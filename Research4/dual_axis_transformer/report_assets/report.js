(function () {
  "use strict";
  var data = window.REPORT_DATA || { summaries: [], resources: null };
  var stageSelect = document.getElementById("stage-select");
  var metricSelect = document.getElementById("metric-select");
  var chart = document.getElementById("chart");
  var table = document.getElementById("summary-table");
  var generatedAt = document.getElementById("generated-at");
  var resourceBox = document.getElementById("resources");
  var svgNS = "http://www.w3.org/2000/svg";

  function unique(values) {
    return Array.from(new Set(values)).sort();
  }

  function addOptions(select, values) {
    select.innerHTML = "";
    values.forEach(function (value) {
      var option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
  }

  function selectedRows() {
    return data.summaries.filter(function (row) {
      return row.stage === stageSelect.value && row.metric === metricSelect.value;
    });
  }

  function svgElement(name, attributes) {
    var element = document.createElementNS(svgNS, name);
    Object.keys(attributes || {}).forEach(function (key) {
      element.setAttribute(key, String(attributes[key]));
    });
    return element;
  }

  function renderChart(rows) {
    chart.innerHTML = "";
    if (!rows.length) {
      chart.textContent = "当前 output 中还没有已完成实验的 final.json。";
      return;
    }
    var width = 900;
    var height = 390;
    var left = 80;
    var right = 30;
    var top = 35;
    var bottom = 85;
    var plotHeight = height - top - bottom;
    var plotWidth = width - left - right;
    var low = Math.min.apply(null, [0].concat(rows.map(function (r) { return r.min; })));
    var high = Math.max.apply(null, [0].concat(rows.map(function (r) { return r.max; })));
    if (low === high) { high = low + 1; }
    function y(value) {
      return top + (high - value) / (high - low) * plotHeight;
    }
    var zero = y(0);
    var slot = plotWidth / rows.length;
    var svg = svgElement("svg", {
      viewBox: "0 0 " + width + " " + height,
      role: "img",
      "aria-label": metricSelect.value + " 方法比较"
    });
    svg.appendChild(svgElement("line", {
      x1: left, y1: zero, x2: width - right, y2: zero, class: "axis"
    }));
    rows.forEach(function (row, index) {
      var center = left + (index + 0.5) * slot;
      var meanY = y(row.mean);
      var bar = svgElement("rect", {
        x: center - Math.min(45, slot * 0.3),
        y: Math.min(meanY, zero),
        width: Math.min(90, slot * 0.6),
        height: Math.max(1, Math.abs(zero - meanY)),
        class: "bar"
      });
      svg.appendChild(bar);
      row.points.forEach(function (point, pointIndex) {
        svg.appendChild(svgElement("circle", {
          cx: center + (pointIndex - (row.points.length - 1) / 2) * 7,
          cy: y(point.value), r: 4, class: "seed-point"
        }));
      });
      var value = svgElement("text", {
        x: center, y: Math.max(top + 14, meanY - 10), class: "value-label"
      });
      value.textContent = Number(row.mean).toPrecision(4);
      svg.appendChild(value);
      var label = svgElement("text", {
        x: center, y: height - bottom + 28, class: "method-label"
      });
      label.textContent = row.method;
      svg.appendChild(label);
    });
    chart.appendChild(svg);
  }

  function renderTable(rows) {
    table.innerHTML = "";
    var header = document.createElement("tr");
    ["方法", "n", "均值", "标准差", "最小", "最大", "各 seed"].forEach(function (name) {
      var th = document.createElement("th");
      th.textContent = name;
      header.appendChild(th);
    });
    table.appendChild(header);
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      var values = [
        row.method, row.n, row.mean, row.std, row.min, row.max,
        row.points.map(function (p) { return p.seed + ":" + Number(p.value).toPrecision(4); }).join(" · ")
      ];
      values.forEach(function (value, index) {
        var td = document.createElement("td");
        td.textContent = index > 1 && index < 6 ? Number(value).toPrecision(5) : value;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
  }

  function renderResources() {
    var resources = data.resources;
    if (!resources) {
      resourceBox.textContent = "尚未生成资源快照。";
      return;
    }
    var cpu = resources.cpu || {};
    var lines = [
      "主机：" + (resources.hostname || "unknown"),
      "CPU：" + (cpu.physical_cores || "?") + " 物理核 / " + (cpu.logical_cores || "?") + " 逻辑核",
      "可用内存：" + (cpu.available_memory_gb == null ? "?" : cpu.available_memory_gb + " GiB")
    ];
    (resources.gpus || []).forEach(function (gpu) {
      lines.push(
        "GPU " + gpu.index + "：" + gpu.name + " · " + gpu.tier +
        " · 空闲 " + gpu.free_memory_gb + "/" + gpu.total_memory_gb + " GiB" +
        (gpu.torch_usable ? " · CUDA 探针通过" : " · CUDA 不可用：" + (gpu.probe_error || "unknown"))
      );
    });
    if (!(resources.gpus || []).length) {
      lines.push("GPU：未检测到可用 CUDA/NVIDIA 设备");
    }
    var list = document.createElement("ul");
    lines.forEach(function (line) {
      var item = document.createElement("li");
      item.textContent = line;
      list.appendChild(item);
    });
    resourceBox.innerHTML = "";
    resourceBox.appendChild(list);
  }

  function refreshMetrics() {
    var metrics = unique(data.summaries.filter(function (row) {
      return row.stage === stageSelect.value;
    }).map(function (row) { return row.metric; }));
    addOptions(metricSelect, metrics.length ? metrics : ["无结果"]);
    refresh();
  }

  function refresh() {
    var rows = selectedRows();
    renderChart(rows);
    renderTable(rows);
  }

  generatedAt.textContent = "生成时间：" + (data.generated_at || "unknown");
  var stages = unique(data.summaries.map(function (row) { return row.stage; }));
  addOptions(stageSelect, stages.length ? stages : [data.stage || "all"]);
  stageSelect.addEventListener("change", refreshMetrics);
  metricSelect.addEventListener("change", refresh);
  refreshMetrics();
  renderResources();
}());
