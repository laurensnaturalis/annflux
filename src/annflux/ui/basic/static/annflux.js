/*
 * Copyright 2025 Naturalis Biodiversity Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
function addDots(
    data,
    x,
    y,
    color_by,
    transform,
    width,
    height,
    d3Element,
    dotsId,
    filters,
    lineThreshold
) {
    if (lineThreshold === undefined) lineThreshold = 5;
    const [show_data, k] = filterData(
        data,
        x,
        y,
        10000,
        transform,
        width,
        height,
        filters,
        "e_0",
        "e_1"
    );

    function getColor(d) {
        const c = (color_by ? d[color_by] : d.color_class);
        if (c === undefined || c === null) {
            return "#888";
        }
        const s = String(c).trim();
        return s.length > 0 ? s : "#888";
    }
    if (localStorage.getItem("debug") === "2") {
        console.log(dotsId, "addDots", filters, x(20));
    }
    // Add dots
    let dotsGroup = d3Element.append("g").attr("id", dotsId);

    const hasAltEmbedding = show_data.length > 0 && show_data[0].e2_0 != null && show_data[0].e2_0 !== undefined && !isNaN(parseFloat(show_data[0].e2_0));

    if (hasAltEmbedding) {
        // Draw connecting lines between primary and alt embedding positions
        dotsGroup
            .selectAll("line.alt-link")
            .data(show_data)
            .enter()
            .append("line")
            .attr("class", "alt-link")
            .attr("x1", (d) => x(d.e_0))
            .attr("y1", (d) => y(d.e_1))
            .attr("x2", (d) => x(parseFloat(d.e2_0)))
            .attr("y2", (d) => y(parseFloat(d.e2_1)))
            .style("stroke", "#aaaaaa")
            .style("stroke-width", 1 / k)
            .style("stroke-opacity", 0.5)
            .style("display", (d) => (lineThreshold !== null && parseFloat(d.alt_line_angle) > lineThreshold) ? null : "none");

        // Draw alt embedding dots
        dotsGroup
            .selectAll("circle.alt-dot")
            .data(show_data)
            .enter()
            .append("circle")
            .attr("class", "alt-dot")
            .attr("cx", (d) => x(parseFloat(d.e2_0)))
            .attr("cy", (d) => y(parseFloat(d.e2_1)))
            .attr("r", 3.0 / k)
            .attr("id", (d) => "dot2-" + d.uid)
            .style("fill", (d) => (d.labeled == 1 ? getColor(d) : "none"))
            .style("stroke", (d) => (d.labeled == 0 ? getColor(d) : "none"))
            .style("stroke-width", 1 / k)
            .style("stroke-dasharray", `${3 / k},${2 / k}`);
    }

    let dots = dotsGroup
        .selectAll("circle.primary-dot")
        .data(show_data)
        .enter()
        .append("circle")
        .attr("class", "primary-dot")
        .attr("cx", function (d) {
            return x(d.e_0);
        })
        .attr("cy", function (d) {
            return y(d.e_1);
        })
        .attr("r", 3.0 / k)
        .attr("id", function (d) {
            return "dot-" + d.uid;
        })
        .style("fill", (d) => (d.labeled == 1 ? getColor(d) : "none"))
        .style("stroke", (d) => (d.labeled == 0 ? getColor(d) : "none"))
        //.style("stroke", function (d) {
        //  return d.color_prob;
        //})
        .style("stroke-width", 1 / k);
    return dots;
}

function addTimeline(data, transform, width, height) {
    const [show_data, k] = filterData(data, x, y, 1000, transform, width, height); // TODO: same data as dots
    // Add dots
    let bars = g
        .append("g")
        .attr("id", "time_bars")
        .selectAll("bar")
        .data(show_data)
        .enter()
        .append("line")
        .attr("x1", function (d) {
            return time_x(d.time_s);
        })
        .attr("x2", function (d) {
            return time_x(d.time_s);
        })
        .attr("y1", function (d) {
            return height;
        })
        .attr("y2", function (d) {
            return height - d.score_predicted * 50;
        })
        .attr("id", function (d) {
            return "time_bar-" + d.time_s;
        })
        .attr("stroke", function (d) {
            return d.color_class;
        })
        .style("stroke-width", function (d) {
            return 1;
        });
    return bars;
}

function filterData(
    data,
    x,
    y,
    max_n,
    transform,
    width,
    height,
    filters,
    xName,
    yName,
    spatialFilterMargin
) {
    console.time("filterData")
    if (localStorage.getItem("debug") === "2") {
        console.log("filterData: transform", transform);
    }
    let hasTransform = transform !== undefined && transform;
    const k = hasTransform ? transform.k : 1;
    const Tx = hasTransform ? transform.x : 1;
    const Ty = hasTransform ? transform.y : 1;
    let show_data_start = [...data];
    //
    for (let [field, operator, value] of filters) {
        if (operator === "equals") {
            const sizeData = show_data_start.length;
            show_data_start = show_data_start.filter((a) => a[field] === value);
            if (localStorage.getItem("debug") === "2") {
                console.log(
                    `|data| = ${sizeData} -> ${[field, operator, value]} -> ${
                        show_data_start.length
                    }`
                );
            }
        } else {
            alert(`unknown operator ${operator}`);
        }
    }

    //TODO(refactor): use filters
    const label_predicted = urlParams.get("label_predicted");
    if (label_predicted) {
        show_data_start = show_data_start.filter((a) =>
            // a["label_predicted"].toLowerCase().includes(label_predicted)
            urlParams.get("not_label_predicted") === "on"
                ? (a["label_predicted"] ?? "").toLowerCase().split(",").indexOf(label_predicted) === -1
                : (a["label_predicted"] ?? "").toLowerCase().split(",").indexOf(label_predicted) > -1
        );
    }
    const label_true = urlParams.get("label_true");
    if (label_true) {
        show_data_start = show_data_start.filter((a) =>
            urlParams.get("not_label_true") === "on"
                ? (a["label_true"] ?? "").toLowerCase().split(",").indexOf(label_true) === -1
                : (a["label_true"] ?? "").toLowerCase().split(",").indexOf(label_true) > -1
        );
    }
    const label_undetermined = urlParams.get("label_undetermined");
    if (label_undetermined) {
        show_data_start = show_data_start.filter((a) =>
            (a["label_undetermined"] ?? "").toLowerCase().includes(label_undetermined)
        );
    }
    let show_data = [];
    // alert(width, height);
    if (spatialFilterMargin === undefined) {
        spatialFilterMargin = 0;
    }
    for (const d of show_data_start) {
        const tX = x(Number(d[xName])) * k + Tx;
        const tY = y(Number(d[yName])) * k + Ty;
        // console.log(tX, tY);
        if (tX > spatialFilterMargin && tX < width && tY > spatialFilterMargin && tY < height) {
            show_data.push(d);
        } else {
            if (localStorage.getItem("debug") >= 3) {
                console.log(`not rendering ${d.uid} ${tX} ${tY}`);
            }
        }
        if (show_data.length >= max_n) {
            break;
        }
    }
    show_data = show_data.reverse();
    if (localStorage.getItem("debug") === "1") {
        console.log("show_data.length", show_data.length);
    }
    // console.log("filterData took", new Date().getTime() / 1000 - now)
    console.timeEnd("filterData")
    return [show_data, k];
}

function getSpacing(show_data, name) {
    let xs = new Set();
    show_data.forEach((el) => xs.add(parseInt(el[name])));
    const sortedX = Array.from(xs).sort((a, b) => a < b);
    return sortedX[0] - sortedX[1];
}

function addImages(
    data,
    x,
    y,
    transform,
    width,
    height,
    d3Element,
    imagesId,
    filters,
    xName,
    yName,
    numberToRender,
    tileSize
) {
    numberToRender = numberToRender ?? 500;

    let tileSizeX, tileSizeY = null;
    let isTileRendering = tileSize === undefined;
    // console.log("show_data[xName]", show_data[xName], xName);
    let [show_data, k] = filterData(
        data,
        x,
        y,
        numberToRender,
        transform,
        width,
        height,
        filters,
        xName,
        yName
    );
    let renderImageType = "thumbnail";
    if (isTileRendering) {
        if (show_data.length > 1) {
            tileSizeX = tileSizeY = x(getSpacing(show_data, xName));
            //tileSizeY = y(getSpacing(show_data, yName));
        } else {
            tileSizeX = tileSizeY = width;
            renderImageType = "original";
        }
    } else {
        tileSizeX = Math.abs(x(tileSize) - x(0)) / Math.sqrt(k);
        tileSizeY = Math.abs(y(tileSize) - y(0)) / Math.sqrt(k);
    }
    //
    [show_data, k] = filterData(
        data,
        x,
        isTileRendering ? x : y,
        numberToRender,
        transform,
        width,
        height,
        filters,
        xName,
        yName,
        isTileRendering ? -2 * tileSizeX : 0,
    );
    // tileSizeY = tileSizeX;
    if (localStorage.getItem("debug") === "2") {
        console.log(imagesId, "addImages", filters, tileSize, tileSizeX, tileSizeY, k, x(1) - x(0));
    }
    images = d3Element
        .append("g")
        .attr("id", imagesId)
        .selectAll("images")
        .data(show_data)
        .enter()
        .append("image")
        .attr("x", function (d) {
            return renderImageType === "thumbnail" ? x(Number(d[xName])) : 0;
        })
        .attr("y", function (d) {
            return renderImageType === "thumbnail" ? (isTileRendering ? x(Number(d[yName])) : y(Number(d[yName]))) : 0; //TODO: think about this
        })
        .attr("width", function (d) {
            return tileSizeX;
        })
        .attr("height", function (d) {
            return tileSizeY;
        })
        .attr("href", function (d) {
            return (d.image_url !== undefined ? d.image_url : (renderImageType === "thumbnail" ? "/images/thumbnail/" : "/images/full/") + d.uid);
        })
        .attr("id", function (d) {
            return "map-" + d.uid; // TODO(refactor): use this map's ID
        })
        .attr("data-label", function (d) {
            return d.label_true;
        });
    //
    // if (localStorage.getItem("debug") >= 2) {
    //     d3Element.selectAll("text").remove();
    //     d3Element
    //         .selectAll("text")
    //         .data(show_data)
    //         .enter()
    //         .append("text")
    //         .attr("x", function (d) {
    //             return renderImageType === "thumbnail" ? x(Number(d[xName])) : 0;
    //         })
    //         .attr("y", function (d) {
    //             return renderImageType === "thumbnail" ? (isTileRendering ? x(Number(d[yName])) : y(Number(d[yName]))) : 0; //TODO: think about this
    //         })
    //         .attr("style", `fill:red;font-size:0.2em`)
    //         .text(function (d) {
    //             return Math.round(renderImageType === "thumbnail" ? (isTileRendering ? x(Number(d[yName])) : y(Number(d[yName]))) : 0);
    //         });
    // }
    //

    d3.selectAll("image").on("click", selectImage);
    return images;
}

function round(num, places) {
    const factor = Math.pow(10, places);
    return Math.round((num + Number.EPSILON) * factor) / factor;
}

function invertMap(childToParentMap) {
    const parentToChildMap = new Map();

    for (const [child, parent] of childToParentMap) {
        if (!parentToChildMap.has(parent)) {
            parentToChildMap.set(parent, []);
        }
        parentToChildMap.get(parent).push(child);
    }

    return parentToChildMap;
}

function renderPerformance(data) {
    console.log("performance");
    const test_performance = data["test_performance"];
    if (test_performance !== undefined) {
        if (test_performance.length > 1) {
            const diff =
                test_performance[test_performance.length - 1][2] -
                test_performance[test_performance.length - 2][2];
            $("#test_performance").html(
                round(test_performance[test_performance.length - 1][2] * 100.0, 1) +
                " (" +
                round(diff * 100.0, 1) +
                ") %"
            );
        }
        $("#percentage_near_labeled").html(
            round(data["percentage_near_labeled"] * 100.0, 1)
        );
        $("#percentage_near_labeled").attr("title",
            `With prediction ${round(data['percentage_labeled_possible'] * 100.0, 1)} %`
        );
    }
}

function excludeDescendants(left, right, parentToChildren, exclusivity) {
    if (parentToChildren.has(left)) {
        for (const child of parentToChildren.get(left)) {
            if (!exclusivity.get(child)) {
                exclusivity.set(child, []);
            }
            for (const el of right) {
                exclusivity.get(child).push(el);
                if (!exclusivity.get(el)) {
                    exclusivity.set(el, []);
                }
                exclusivity.get(el).push(child);
            }
        }
    }
}

controlHtml = `<div id="map_control">
      <table>
      <tr>
          <td>Status <img src="/static/loading_circle.gif" width="58px" id="loading_circle" style="display: none;"/></td>
          <td>
            <span id="status"></span>
            <div id="status" style="width:100%; border: 1px solid white; display: flex; position: relative; height:1em">
              <div id="progress" style="background-color:darkgreen; width:100%; position: absolute; height: 100%; bottom:0; top:0"></div>
            </div>
          </td>
        </tr>
        <tr>
          <td><img src="/static/annflux.svg" width="32px"/></td>
          <td><span id="speed_active_time"></span> active, <span id="speed_annotation"></span> <img src="/static/label.svg" width="24px"/>/h, likely certain <span id="speed_certain"></span><img src="/static/label.svg" width="24px"/>/h</td>        
        <tr>
          <td>Training</td>
          <td>
            <a href="javascript:void(0)" onclick="forceLinearTrain()"
              >Linear</a
            > 
            <a href="javascript:void(0)" onclick="groupTrain()" id="groupTrainButton"
              >Group</a
            >
          </td>
        </tr>        
        <tr>
          <td>
            <a href="/detailed_performance">Performance</a>
          </td>
          <td>
            acc. = <span id="test_performance"></span> aP = <span id="average_precision" title="Average precision: averaged across labels how often a prediction is correct for a certain label"></span> % - aR <span id="average_recall"></span> %
          </td>
        </tr>
        <tr>
          <td>Space covered</td>
          <td>
            <span id="percentage_near_labeled"></span> %
          </td>
        </tr>
        <tr>
          <td>Likely certain</td>
          <td>
            <span id="likely_certain_perc"></span> %
          </td>
        </tr>
        <tr>
          <td>Likely certain (unlabeled)</td>
          <td>
            <span id="likely_certain_perc_unlabeled"></span> %
          </td>
        </tr>
        <tr>
          <td>UI log</td>
          <td>
            <span id="ui_log_last"></span>
          </td>
        </tr>

      </table>
      <a href="#" onclick="persistentToggle('view_config')">View <img src="/static/settings.svg" width="16px"/></a> <a href="/annflux">Reset</a>
      <table id="view_config">

        <tr>
          <td width="300px">Ranking</td>
          <td>
            <select id="as_ranking_column" onchange="changeOption(this)">
              <option value="score_predicted">
                Prediction uncertainty
              </option>
              <option value="high_label_entropy">High label entropy</option>
              <option value="score_true">True probability</option>
              <option value="fre">FRE</option>
              <option value="fre_strat">FRE - stratified</option>
              <option value="certain_incorrect">Certain incorrect</option>
              <option value="most_needed" selected="selected">Most needed</option>
              <option value="incorrect_score">Incorrect score</option></select
            ><input
              type="checkbox"
              id="invert_ranking"
              name="invert_ranking"
              onchange="changeOption(this)"
            />
            <label for="invert_ranking">INVERT</label>
          </td>
        </tr>
        <tr>
          <td>Label predicted</td>
          <td>
            <select id="label_predicted" onchange="changeOption(this)"></select>
            <input
              type="checkbox"
              id="not_label_predicted"
              name="not_label_true"
              onchange="changeOption(this)"
            />
            <label for="not_label_predicted">NOT</label>
          </td>
        </tr>
        <tr>
          <td>Label true</td>
          <td>
            <select id="label_true" onchange="changeOption(this)"></select>
            <input
              type="checkbox"
              id="not_label_true"
              name="not_label_true"
              onchange="changeOption(this)"
            />
            <label for="not_label_true">NOT</label>
          </td>
        </tr>
        <tr>
          <td>Label undetermined</td>
          <td>
            <select
              id="label_undetermined"
              onchange="changeOption(this)"
            ></select>
          </td>
        </tr>
        <tr>
          <td>Label ignore</td>
          <td>
            <select
              id="label_ignore"
              onchange="changeOption(this)"
            ></select>
          </td>
        </tr>
        <tr>
          <td>Filter</td>
          <td>
            <input type="text"
              id="filter_query"
            /> <a href="#" onclick="changeOption(document.getElementById('filter_query'))">Apply</a>
          </td>
        </tr>
        <tr>
          <td>Show (un)labeled</td>
          <td>
            <select id="show_labeled" onchange="changeOption(this)">
              <option value="unlabeled" selected="selected">unlabeled</option>
              <option value="labeled">labeled</option>
              <option value="both">both</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>Color</td>
          <td>
            <select id="color_map" onchange="changeOption(this)">
              <option value="color_class" selected="selected">
                Predicted class
              </option>
              <option value="color_prob">Probability</option>
              <option value="color_fre">FRE</option>
              <option value="dp_cluster_color">Density peak cluster</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>Num in gallery</td>
          <td>
            <select id="num_in_gallery" onchange="changeOption(this)">
              <option value="10" selected="selected">10</option>
              <option value="20">20</option>
              <option value="50">50</option>
              <option value="100">100</option>
              <option value="200">200</option>
              <option value="500">500</option>
            </select>
          </td>
        </tr>
        <tr>
          <td>
            <label for="ignore_double_checked">Ignore double-checked</label>
          </td>
          <td>
            <input
              type="checkbox"
              id="ignore_double_checked"
              name="ignore_double_checked"
              onchange="changeOption(this)"
            />
          </td>
        </tr>
      </table>
    </div>`;

const mapHtml = `<div id="my_dataviz" tabindex="0"></div>`;






// Example map (object) to render
const myMap = {
    "Name": "Laurens Hogeweg",
    "Country": "Netherlands",
    "Language": "JavaScript",
    "Framework": "React"
};

// Function to render the map as a table inside a div
function renderMapAsTable(map, containerId) {
    // Create the table structure
    const $table = $('<table>').css('border', '1px solid black');

    // Add table header
    const $thead = $('<thead>');
    const $headerRow = $('<tr>');
    $headerRow.append($('<th>').text('Key'));
    $headerRow.append($('<th>').text('Value'));
    $thead.append($headerRow);
    $table.append($thead);

    // Add table body
    const $tbody = $('<tbody>');
    $.each(map, function(key, value) {
        const $row = $('<tr>');
        $row.append($('<td>').text(key));
        $row.append($('<td>').text(value));
        $tbody.append($row);
    });
    $table.append($tbody);

    // Append the table to the specified div
    $(`#${containerId}`).html($table);
}
