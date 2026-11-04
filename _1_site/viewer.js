const fields = ['team', 'player', 'year', 'month', 'day'];
const status = document.querySelector('#status'), reset = document.querySelector('#reset');
const gameDirections = ['prior', 'next'];
function gameElements(direction) {
  return Object.fromEntries(['title', 'status', 'empty', 'heatmap', 'statistics', 'file'].map(name => [name, document.querySelector(`#${direction}-${name}`)]));
}
let manifest, controller, generation = 0;
const selects = Object.fromEntries(fields.map(field => {
  const container = document.createElement('div'), label = document.createElement('label'), select = document.createElement('select');
  label.htmlFor = field; label.textContent = field[0].toUpperCase() + field.slice(1);
  select.id = field; select.disabled = true;
  select.addEventListener('change', () => changed(fields.indexOf(field)));
  container.append(label, select); document.querySelector('#filters').append(container);
  return [field, select];
}));
function message(element, text, error = false) { element.textContent = text; element.classList.toggle('error', error); }
async function fetchJSON(url, signal) {
  const response = await fetch(url, {signal});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}
function populate(index) {
  const rows = manifest.filter(row => fields.slice(0, index).every(key => row[key] === selects[key].value));
  const field = fields[index], select = selects[field];
  const values = [...new Set(rows.map(row => row[field]))].sort((a,b) => a.localeCompare(b, undefined, {numeric:true}));
  select.replaceChildren(new Option(`Select ${field}…`, ''), ...values.map(value => new Option(field === 'team' ? `Team ${value}` : value, value)));
  select.disabled = false;
}
function clearPanes() {
  reset.disabled = true;
  const oldPlot = document.querySelector('#plot');
  if (window.Plotly) Plotly.purge(oldPlot);
  const newPlot = document.createElement('div'); newPlot.id = 'plot';
  newPlot.setAttribute('aria-label', 'Interactive 3D player trajectory'); oldPlot.replaceWith(newPlot);
  document.querySelector('#arc-title').textContent = 'Player trajectory';
  document.querySelector('#file').textContent = '';
  const scale = document.querySelector('#shared-colorbar'); scale.hidden = true;
  for (const direction of gameDirections) {
    const pane = gameElements(direction);
    pane.title.textContent = `${direction === 'prior' ? 'Prior' : 'Next'} game · Acceleration heatmap`;
    pane.file.textContent = '';
    pane.statistics.replaceChildren();
    pane.heatmap.hidden = true; pane.heatmap.removeAttribute('src');
    pane.empty.hidden = false; pane.empty.textContent = 'No session selected.';
    message(pane.status, `Select a session to find the ${direction} game.`);
  }
}
function showStatistics(reports, box) {
  box.replaceChildren();
  const heading = document.createElement('h3'); heading.textContent = 'Game performance';
  const note = document.createElement('p'); note.className = 'stats-note';
  note.textContent = reports.length === 0 ? 'No game statistics available.' :
    (reports.some(report => report.selected_player) ? 'Self-reported ratings / 10' : 'No player report. Teammate ratings / 10');
  box.append(heading, note);
  for (const report of reports) {
    const entry = document.createElement('article'); entry.className = report.selected_player ? 'report selected' : 'report';
    const player = document.createElement('div'); player.className = 'report-player';
    const uuid = report.player_name.slice(report.player_name.indexOf('-') + 1);
    player.textContent = report.selected_player ? 'Selected player' : `Player ${uuid.slice(0, 8)}`;
    player.title = report.player_name;
    const scores = document.createElement('dl'); scores.className = 'score-grid';
    for (const [label, value] of [['Team', report.team_performance], ['Offense', report.offensive_performance], ['Defense', report.defensive_performance]]) {
      const group = document.createElement('div'), term = document.createElement('dt'), score = document.createElement('dd');
      term.textContent = label; score.textContent = String(value);
      group.append(term, score); scores.append(group);
    }
    entry.append(player, scores); box.append(entry);
  }
}
async function loadArc(params, signal, id, plot) {
  message(status, 'Reading session and calculating trajectory…');
  try {
    if (!window.Plotly) throw new Error('Local Plotly library did not load');
    const track = await fetchJSON('/api/session?' + params, signal);
    if (id !== generation) return;
    document.querySelector('#arc-title').textContent = `Player trajectory · ${track.is_game_day ? 'Game day' : 'Non-game session'}`;
    const xSpan = span(track.x), ySpan = span(track.y);
    const horizontalSpan = Math.max(xSpan, ySpan);
    await Plotly.newPlot(plot, [{type:'scatter3d', mode:'lines', x:track.x, y:track.y, z:track.z,
      line:{width:6, color:track.color, colorscale:[[0,'lightblue'],[1,'darkblue']], cmin:0, cmax:1, showscale:true, colorbar:{title:{text:'Normalized, Smoothed<br>Acceleration'}, thickness:12,len:0.55}},
      customdata:track.hover.map((index, i) => [index, track.latitude[i], track.longitude[i]]),
      hovertemplate:'east: %{x:.2f} m<br>north: %{y:.2f} m<br>lat: %{customdata[1]:.6f}<br>lon: %{customdata[2]:.6f}<br>index: %{customdata[0]}<extra></extra>'
    }], {scene:{xaxis:{title:{text:'East (m)'}}, yaxis:{title:{text:'North (m)'}}, zaxis:{title:{text:'Time (normalized)'}},
      aspectmode:'manual', aspectratio:{x:xSpan/horizontalSpan,y:ySpan/horizontalSpan,z:1}}, margin:{l:0,r:0,b:0,t:15}, font:{family:'system-ui',color:'#344b65'}}, {responsive:true,displaylogo:false});
    if (id !== generation) return;
    reset.disabled = false;
    document.querySelector('#file').textContent = track.file;
    message(status, `${track.date} · ${track.rows.toLocaleString()} samples · ${track.x.length.toLocaleString()} points. Drag to rotate; scroll to zoom.`);
  } catch (error) { if (id === generation && error.name !== 'AbortError') message(status, error.message, true); }
}
function span(values) {
  let min = Infinity, max = -Infinity;
  for (const value of values) { min = Math.min(min, value); max = Math.max(max, value); }
  return max - min;
}
async function loadGames(params, signal, id) {
  for (const direction of gameDirections) {
    const pane = gameElements(direction);
    message(pane.status, `Finding the ${direction} game within 2 weeks…`); pane.empty.textContent = 'Loading…';
  }
  try {
    const response = await fetchJSON('/api/games?' + params, signal);
    if (id !== generation) return;
    for (const direction of gameDirections) {
      const pane = gameElements(direction), result = response[direction];
      if (result.game === null) { message(pane.status, result.message); pane.empty.textContent = result.message; continue; }
      const game = result.game;
      const days = direction === 'prior' ? game.days_before : game.days_after;
      pane.title.textContent = `${direction === 'prior' ? 'Prior' : 'Next'} game · ${game.date}`;
      message(pane.status, `${days} day${days === 1 ? '' : 's'} ${direction === 'prior' ? 'before' : 'after'} · ${game.rows.toLocaleString()} samples`);
      pane.heatmap.src = game.image; pane.heatmap.hidden = false; pane.empty.hidden = true;
      pane.file.textContent = game.file; pane.file.title = game.file;
      showStatistics(game.statistics, pane.statistics);
    }
    if (response.scale !== null) {
      const scale = document.querySelector('#shared-colorbar');
      scale.title = `Shared smoothed acceleration range: ${response.scale.min.toFixed(4)}–${response.scale.max.toFixed(4)}`;
      scale.hidden = false;
    }
  } catch (error) {
    if (id === generation && error.name !== 'AbortError') {
      for (const direction of gameDirections) {
        const pane = gameElements(direction); message(pane.status, error.message, true); pane.empty.textContent = 'Unable to load game data.';
      }
    }
  }
}
async function changed(index) {
  const id = ++generation;
  if (controller) controller.abort();
  clearPanes();
  for (const field of fields.slice(index + 1)) { selects[field].replaceChildren(new Option(`Select ${field}…`, '')); selects[field].disabled = true; }
  if (!selects[fields[index]].value) { message(status, `Select ${fields[index]} to continue.`); return; }
  if (index < fields.length - 1) { populate(index + 1); message(status, `Select ${fields[index + 1]} to continue.`); return; }
  controller = new AbortController();
  const params = new URLSearchParams(fields.map(key => [key, selects[key].value]));
  await Promise.all([loadArc(params, controller.signal, id, document.querySelector('#plot')), loadGames(params, controller.signal, id)]);
}
reset.onclick = () => Plotly.relayout('plot', {'scene.camera':{eye:{x:1.25,y:1.25,z:1.25},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}}});
try {
  manifest = await fetchJSON('/api/manifest');
  fields.forEach(field => selects[field].replaceChildren(new Option(`Select ${field}…`, '')));
  populate(0); message(status, 'Select team to begin.');
} catch (error) { message(status, error.message, true); }
