/* =========================================================================
   Decision Debugger — frontend controller (vanilla JS, no build step)
   Talks only to same-origin /api/* endpoints. Never handles an API key.
   All API/LLM text is inserted via textContent or the safe markdown renderer.
   ========================================================================= */
(function () {
  "use strict";

  // ---------------------------------------------------------------- state
  var state = {
    sessionId: null,
    question: null,
  };

  // ---------------------------------------------------------------- dom refs
  var screens = {
    setup: document.getElementById("screen-setup"),
    question: document.getElementById("screen-question"),
    result: document.getElementById("screen-result"),
  };
  var el = {
    setupForm: document.getElementById("setup-form"),
    contextInput: document.getElementById("context-input"),
    startBtn: document.getElementById("start-btn"),
    setupError: document.getElementById("setup-error"),
    loading: document.getElementById("loading"),
    loadingText: document.getElementById("loading-text"),
    questionBody: document.getElementById("question-body"),
    progressFill: document.getElementById("progress-fill"),
    progressLabel: document.getElementById("progress-label"),
    restartBtn: document.getElementById("restart-btn"),
    // result
    resultWinner: document.getElementById("result-winner"),
    robustnessBadge: document.getElementById("robustness-badge"),
    marginBadge: document.getElementById("margin-badge"),
    report: document.getElementById("report"),
    driversBlock: document.getElementById("drivers-block"),
    driversList: document.getElementById("drivers-list"),
    conflictBlock: document.getElementById("conflict-block"),
    conflictList: document.getElementById("conflict-list"),
    rankingBlock: document.getElementById("ranking-block"),
    rankingList: document.getElementById("ranking-list"),
    nextinfoBlock: document.getElementById("nextinfo-block"),
    nextinfoList: document.getElementById("nextinfo-list"),
  };

  // ---------------------------------------------------------------- helpers
  function showScreen(name) {
    Object.keys(screens).forEach(function (key) {
      var node = screens[key];
      var active = key === name;
      node.hidden = !active;
      node.classList.toggle("screen--active", active);
    });
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  function escapeHtml(str) {
    var s = str == null ? "" : String(str);
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* Minimal SAFE markdown renderer.
     Strategy: escape EVERYTHING first, then re-introduce a tiny allowlist
     (**bold**, "- " bullets, blank-line paragraphs, single line breaks).
     No raw HTML from the API ever reaches the DOM. */
  function renderMarkdown(md) {
    var text = (md == null ? "" : String(md)).replace(/\r\n/g, "\n").trim();
    if (!text) return "";

    var blocks = text.split(/\n{2,}/); // paragraphs / lists separated by blank line
    var html = "";

    blocks.forEach(function (block) {
      var lines = block.split("\n");
      var isList = lines.every(function (l) {
        return /^\s*[-*]\s+/.test(l) || l.trim() === "";
      });

      if (isList && lines.some(function (l) { return /^\s*[-*]\s+/.test(l); })) {
        html += "<ul>";
        lines.forEach(function (l) {
          if (l.trim() === "") return;
          var item = l.replace(/^\s*[-*]\s+/, "");
          html += "<li>" + inline(item) + "</li>";
        });
        html += "</ul>";
      } else {
        html += "<p>" + inline(block).replace(/\n/g, "<br />") + "</p>";
      }
    });

    return html;
  }

  // inline formatting on already-escaped text: only **bold**
  function inline(raw) {
    var safe = escapeHtml(raw);
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    return safe;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function makeEl(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function pct(x) {
    var n = Math.max(0, Math.min(1, Number(x) || 0));
    return (n * 100).toFixed(1) + "%";
  }

  // ---------------------------------------------------------------- API
  function apiPost(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(handleJson);
  }
  function apiGet(path) {
    return fetch(path).then(handleJson);
  }
  function handleJson(res) {
    return res.text().then(function (raw) {
      var data = null;
      try {
        data = raw ? JSON.parse(raw) : null;
      } catch (e) {
        data = null;
      }
      if (!res.ok) {
        var detail = data && data.detail ? data.detail : "요청을 처리하지 못했습니다.";
        var err = new Error(detail);
        err.detail = detail;
        throw err;
      }
      return data;
    });
  }

  // ---------------------------------------------------------------- loading
  var LOADING_MESSAGES = [
    "상황을 읽고 있어요",
    "선택지를 정리하는 중",
    "판단 기준을 세우는 중",
    "각 선택지를 채점하는 중",
    "가장 좋은 질문을 고르는 중",
  ];
  var loadingTimer = null;
  function startLoading() {
    var i = 0;
    el.loadingText.textContent = LOADING_MESSAGES[0];
    el.loading.hidden = false;
    loadingTimer = window.setInterval(function () {
      i = (i + 1) % LOADING_MESSAGES.length;
      el.loadingText.textContent = LOADING_MESSAGES[i];
    }, 4200);
  }
  function stopLoading() {
    if (loadingTimer) {
      window.clearInterval(loadingTimer);
      loadingTimer = null;
    }
    el.loading.hidden = true;
  }

  // ---------------------------------------------------------------- SETUP
  el.setupForm.addEventListener("submit", function (event) {
    event.preventDefault();
    var context = (el.contextInput.value || "").trim();
    el.setupError.hidden = true;
    if (context.length < 5) {
      el.setupError.textContent = "고민 내용을 조금 더 자세히 적어주세요.";
      el.setupError.hidden = false;
      return;
    }

    el.startBtn.disabled = true;
    startLoading();

    apiPost("/api/sessions", { context: context })
      .then(function (data) {
        state.sessionId = data.session && data.session.id;
        if (!state.sessionId) throw new Error("세션을 생성하지 못했습니다.");
        if (data.done || !data.question) {
          return loadResult();
        }
        state.question = data.question;
        stopLoading();
        renderQuestion(data.question);
        showScreen("question");
      })
      .catch(function (err) {
        stopLoading();
        el.setupError.textContent = err.detail || err.message || "분석 중 오류가 발생했습니다. 다시 시도해주세요.";
        el.setupError.hidden = false;
      })
      .finally(function () {
        el.startBtn.disabled = false;
      });
  });

  // ---------------------------------------------------------------- QUESTIONING
  function renderQuestion(q) {
    state.question = q;
    // progress
    var answered = (q.progress && q.progress.answered) || 0;
    var max = (q.progress && q.progress.max) || 8;
    var ratio = max > 0 ? Math.min(1, (answered + 1) / max) : 0;
    el.progressFill.style.width = (ratio * 100).toFixed(0) + "%";
    el.progressLabel.textContent = "질문 " + (answered + 1) + " / " + max;

    clear(el.questionBody);

    // Prompt line above the cards. Card comparisons share one consistent
    // instruction; any indicator question falls back to its own text.
    var promptText =
      q.kind === "weight_pairwise" || q.kind === "sub_question"
        ? "이번 결정에서 더 중요한 쪽을 골라주세요"
        : q.question || "";
    if (promptText) {
      el.questionBody.appendChild(makeEl("p", "question__prompt", promptText));
    }

    if (q.kind === "weight_pairwise") {
      renderPairwise(q);
    } else if (q.kind === "sub_question") {
      renderSubQuestion(q);
    } else {
      renderIndicator(q);
    }
  }

  function renderSubQuestion(q) {
    // Same two-card layout as a normal question, so the format is uniform.
    renderCards(q, Array.isArray(q.cards) ? q.cards : []);
  }

  // Shared two-card renderer — every question (pairwise AND decomposed sub-question)
  // uses this exact layout: two tappable cards (name + example) + "잘 모르겠어요".
  function renderCards(q, cards) {
    var grid = makeEl("div", "pair");
    cards.forEach(function (c) {
      var card = makeEl("button", "pair-card");
      card.type = "button";
      card.appendChild(makeEl("span", "pair-card__name", c.name || ""));
      if (c.example) {
        card.appendChild(makeEl("span", "pair-card__example", c.example));
      }
      card.addEventListener("click", function () {
        submit(q.id, c.value);
      });
      grid.appendChild(card);
    });
    el.questionBody.appendChild(grid);

    var secondary = makeEl("div", "pair-secondary");
    var unknown = makeEl("button", "btn btn--quiet", "잘 모르겠어요");
    unknown.type = "button";
    unknown.addEventListener("click", function () {
      submit(q.id, "unknown");
    });
    secondary.appendChild(unknown);
    el.questionBody.appendChild(secondary);
  }

  function renderPairwise(q) {
    renderCards(q, [
      { name: (q.factor_a && q.factor_a.name) || "", example: q.example_a, value: "a" },
      { name: (q.factor_b && q.factor_b.name) || "", example: q.example_b, value: "b" },
    ]);
  }

  function renderIndicator(q) {
    var type = q.answer_type || "binary";

    if (type === "binary") {
      var row = makeEl("div", "answer-row");
      [["네", "yes"], ["아니오", "no"]].forEach(function (spec) {
        var b = makeEl("button", "answer-pill", spec[0]);
        b.type = "button";
        b.addEventListener("click", function () {
          submit(q.id, spec[1]);
        });
        row.appendChild(b);
      });
      el.questionBody.appendChild(row);
      return;
    }

    if (type === "choice") {
      var crow = makeEl("div", "answer-row");
      var choices = Array.isArray(q.choices) ? q.choices : [];
      choices.forEach(function (c) {
        var b = makeEl("button", "answer-pill", c.label != null ? c.label : c.value);
        b.type = "button";
        b.addEventListener("click", function () {
          submit(q.id, c.value);
        });
        crow.appendChild(b);
      });
      el.questionBody.appendChild(crow);
      return;
    }

    if (type === "scale") {
      var scale = q.scale || { min: 1, max: 5 };
      var min = Number(scale.min);
      var max = Number(scale.max);
      if (!isFinite(min)) min = 1;
      if (!isFinite(max) || max < min) max = min + 4;

      var seg = makeEl("div", "scale-row");
      for (var v = min; v <= max; v++) {
        (function (val) {
          var b = makeEl("button", "scale-seg", String(val));
          b.type = "button";
          b.addEventListener("click", function () {
            submit(q.id, val);
          });
          seg.appendChild(b);
        })(v);
      }
      var wrap = makeEl("div");
      wrap.style.display = "flex";
      wrap.style.flexDirection = "column";
      wrap.style.alignItems = "center";
      wrap.appendChild(seg);
      var ends = makeEl("div", "scale-ends");
      ends.appendChild(makeEl("span", null, "전혀 아님"));
      ends.appendChild(makeEl("span", null, "매우 그럼"));
      wrap.appendChild(ends);
      el.questionBody.appendChild(wrap);
      return;
    }

    // count
    var crow2 = makeEl("div", "count-row");
    var input = makeEl("input", "count-input");
    input.type = "number";
    input.min = "0";
    input.step = "1";
    input.setAttribute("aria-label", "숫자 입력");
    input.placeholder = "0";
    var confirm = makeEl("button", "btn btn--primary", "확인");
    confirm.type = "button";
    function sendCount() {
      var raw = input.value.trim();
      if (raw === "") {
        input.focus();
        return;
      }
      var n = parseInt(raw, 10);
      if (isNaN(n) || n < 0) {
        input.focus();
        return;
      }
      submit(q.id, n);
    }
    confirm.addEventListener("click", sendCount);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        sendCount();
      }
    });
    crow2.appendChild(input);
    crow2.appendChild(confirm);
    el.questionBody.appendChild(crow2);
    input.focus();
  }

  // disable interactive controls while a submit is in flight
  function setQuestionBusy(busy) {
    var controls = el.questionBody.querySelectorAll("button, input");
    for (var i = 0; i < controls.length; i++) {
      controls[i].disabled = busy;
    }
  }

  function submit(questionId, value) {
    if (!state.sessionId) return;
    setQuestionBusy(true);
    apiPost("/api/sessions/" + encodeURIComponent(state.sessionId) + "/answers", {
      question_id: questionId,
      value: value,
    })
      .then(function (data) {
        if (data.done || !data.question) {
          return loadResult();
        }
        renderQuestion(data.question);
      })
      .catch(function (err) {
        setQuestionBusy(false);
        // gentle inline notice — keep it calm, not error-looking
        el.progressLabel.textContent = (err.detail || err.message || "다시 시도해주세요.");
      });
  }

  // ---------------------------------------------------------------- RESULT
  function loadResult() {
    startLoading();
    el.loadingText.textContent = "결론을 정리하는 중";
    return apiGet("/api/sessions/" + encodeURIComponent(state.sessionId) + "/result")
      .then(function (data) {
        stopLoading();
        renderResult(data);
        showScreen("result");
      })
      .catch(function (err) {
        stopLoading();
        // fall back to question screen label rather than a hard error wall
        el.progressLabel.textContent = (err.detail || err.message || "리포트를 불러오지 못했습니다.");
        showScreen("question");
      });
  }

  function renderResult(data) {
    data = data || {};

    // winner
    var winner = data.winner || {};
    el.resultWinner.textContent = winner.name || "결론";

    // robustness badge
    var robust = data.robustness === "close" ? "close" : "stable";
    el.robustnessBadge.textContent = robust === "stable" ? "탄탄함" : "근소함";
    el.robustnessBadge.className = "badge badge--" + robust;

    // margin badge (optional, quiet)
    if (typeof data.margin === "number" && isFinite(data.margin)) {
      el.marginBadge.hidden = false;
      el.marginBadge.textContent = "격차 " + (data.margin * 100).toFixed(1) + "%";
    } else {
      el.marginBadge.hidden = true;
    }

    // report (safe markdown)
    el.report.innerHTML = renderMarkdown(data.report || "");

    // drivers
    renderDrivers(Array.isArray(data.drivers) ? data.drivers : []);

    // conflict
    renderConflict(Array.isArray(data.conflict) ? data.conflict : []);

    // ranking
    renderRanking(Array.isArray(data.ranking) ? data.ranking : []);

    // next info
    renderNextInfo(Array.isArray(data.next_info) ? data.next_info : []);
  }

  function renderDrivers(drivers) {
    clear(el.driversList);
    if (!drivers.length) {
      el.driversBlock.hidden = true;
      return;
    }
    el.driversBlock.hidden = false;

    var maxWeight = drivers.reduce(function (m, d) {
      return Math.max(m, Number(d.weight) || 0);
    }, 0) || 1;

    drivers.forEach(function (d) {
      var li = makeEl("li", "driver");

      var head = makeEl("div", "driver__head");
      head.appendChild(makeEl("span", "driver__name", d.factor || ""));
      head.appendChild(makeEl("span", "driver__weight", "중요도 " + pct(d.weight)));
      li.appendChild(head);

      var track = makeEl("div", "bar-track");
      var fill = makeEl("div", "bar-fill");
      // bar length is relative to the strongest driver so differences read clearly
      var rel = (Number(d.weight) || 0) / maxWeight;
      fill.style.width = (rel * 100).toFixed(1) + "%";
      track.appendChild(fill);

      // flip threshold annotation (absolute weight position on a 0..1 axis,
      // but axis here is the relative track, so place against the same scale)
      if (d.flip_threshold != null && isFinite(d.flip_threshold)) {
        var flipRel = (Number(d.flip_threshold) || 0) / maxWeight;
        flipRel = Math.max(0, Math.min(1, flipRel));
        var marker = makeEl("div", "bar-flip");
        marker.style.left = (flipRel * 100).toFixed(1) + "%";
        marker.title = "결론이 뒤집히는 지점";
        track.appendChild(marker);
      }
      li.appendChild(track);

      if (d.flip_threshold != null && isFinite(d.flip_threshold)) {
        var note = makeEl("p", "driver__flip");
        note.appendChild(document.createTextNode("중요도가 "));
        var b = makeEl("strong", null, pct(d.flip_threshold));
        note.appendChild(b);
        note.appendChild(document.createTextNode(" 근처가 되면 결론이 뒤집힐 수 있어요."));
        li.appendChild(note);
      } else {
        var stableNote = makeEl("p", "driver__flip", "이 기준만으로는 결론이 쉽게 뒤집히지 않아요.");
        li.appendChild(stableNote);
      }

      el.driversList.appendChild(li);
    });
  }

  function renderConflict(conflict) {
    clear(el.conflictList);
    if (!conflict.length) {
      el.conflictBlock.hidden = true;
      return;
    }
    el.conflictBlock.hidden = false;

    conflict.forEach(function (c) {
      var li = makeEl("li", "conflict");
      li.appendChild(makeEl("div", "conflict__factor", c.factor || ""));

      var row = makeEl("div", "conflict__row");
      var stated = makeEl("span");
      stated.appendChild(document.createTextNode("처음 생각 "));
      stated.appendChild(makeEl("b", null, pct(c.stated)));
      var revealed = makeEl("span");
      revealed.appendChild(document.createTextNode("실제 선택 "));
      revealed.appendChild(makeEl("b", null, pct(c.revealed)));
      row.appendChild(stated);
      row.appendChild(revealed);
      li.appendChild(row);

      var dirText =
        c.direction === "up"
          ? "실제로는 이 기준을 처음 생각보다 더 중요하게 여기고 있어요."
          : c.direction === "down"
          ? "실제로는 이 기준을 처음 생각보다 덜 중요하게 여기고 있어요."
          : "처음 생각과 실제 선택이 어긋나는 기준이에요.";
      li.appendChild(makeEl("p", "conflict__dir", dirText));

      el.conflictList.appendChild(li);
    });
  }

  function renderRanking(ranking) {
    clear(el.rankingList);
    if (!ranking.length) {
      el.rankingBlock.hidden = true;
      return;
    }
    el.rankingBlock.hidden = false;

    var maxUtil = ranking.reduce(function (m, r) {
      return Math.max(m, Number(r.utility) || 0);
    }, 0) || 1;

    ranking.forEach(function (r, idx) {
      var li = makeEl("li", "rank-item" + (idx === 0 ? " rank-item--lead" : ""));

      var head = makeEl("div", "rank-item__head");
      head.appendChild(makeEl("span", "rank-item__name", (idx + 1) + ". " + (r.name || "")));
      head.appendChild(makeEl("span", "rank-item__util", (Number(r.utility) || 0).toFixed(3)));
      li.appendChild(head);

      var track = makeEl("div", "bar-track");
      var fill = makeEl("div", "bar-fill");
      var rel = (Number(r.utility) || 0) / maxUtil;
      fill.style.width = (Math.max(0, Math.min(1, rel)) * 100).toFixed(1) + "%";
      track.appendChild(fill);
      li.appendChild(track);

      el.rankingList.appendChild(li);
    });
  }

  function renderNextInfo(items) {
    clear(el.nextinfoList);
    if (!items.length) {
      el.nextinfoBlock.hidden = true;
      return;
    }
    el.nextinfoBlock.hidden = false;
    items.forEach(function (text) {
      el.nextinfoList.appendChild(makeEl("li", null, text));
    });
  }

  // ---------------------------------------------------------------- restart
  el.restartBtn.addEventListener("click", function () {
    state.sessionId = null;
    state.question = null;
    el.contextInput.value = "";
    el.setupError.hidden = true;
    el.startBtn.disabled = false;
    showScreen("setup");
    el.contextInput.focus();
  });

  // ---------------------------------------------------------------- init
  showScreen("setup");
})();
