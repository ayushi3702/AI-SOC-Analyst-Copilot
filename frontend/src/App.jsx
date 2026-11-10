import { useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function postJson(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return body;
}

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return body;
}

function formatTime(event, source) {
  if (source === "lanl_user_hour_features") {
    return `Relative hour ${event.hour_index}`;
  }
  return new Date(event.observed_at.replace(" ", "T")).toLocaleString();
}

function EventFacts({ event, source }) {
  if (source === "lanl_user_hour_features") {
    return (
      <div className="facts">
        <span>user {event.user_id}</span>
        <span>{event.event_count} auth events</span>
        <span>{event.success_count} successes</span>
        <span>{event.failure_count} failures</span>
        <span>{event.previous_24h_events} events in prior 24h</span>
      </div>
    );
  }
  return (
    <div className="facts">
      <span>{event.protocol}</span>
      <span>{event.src_ip}:{event.src_port}</span>
      <span>→</span>
      <span>{event.dst_ip}:{event.dst_port}</span>
      <span>{event.fwd_packets} / {event.bwd_packets} packets</span>
    </div>
  );
}

function App() {
  const [source, setSource] = useState("network");
  const [form, setForm] = useState({
    incident_id: "demo-001",
    entity: "192.168.10.5",
    start_time: "2017-03-07T08:55:00",
    end_time: "2017-03-07T09:05:00",
    user_id: "U748@DOM1",
    start_hour: "0",
    end_hour: "1391",
    limit: "100",
    src_ip: "",
    dst_ip: "",
    source_port: "",
    destination_port: "",
    protocol: "",
  });
  const [result, setResult] = useState(null);
  const [report, setReport] = useState(null);
  const [decisions, setDecisions] = useState({});
  const [decisionHistory, setDecisionHistory] = useState([]);
  const [analyst, setAnalyst] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const events = result?.evidence || [];
  const timeline = useMemo(
    () => [...events].sort((a, b) => {
      const left = source === "network" ? a.observed_at : a.hour_index;
      const right = source === "network" ? b.observed_at : b.hour_index;
      return left < right ? -1 : left > right ? 1 : 0;
    }),
    [events, source],
  );

  function update(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function networkPayload() {
    const payload = {
      incident_id: form.incident_id,
      entity: form.entity,
      start_time: form.start_time,
      end_time: form.end_time,
      limit: Number(form.limit),
    };
    for (const field of ["src_ip", "dst_ip", "protocol"]) {
      if (form[field].trim()) payload[field] = form[field].trim();
    }
    for (const field of ["source_port", "destination_port"]) {
      if (form[field] !== "") payload[field] = Number(form[field]);
    }
    return payload;
  }

  async function search() {
    setBusy(true);
    setError("");
    setReport(null);
    try {
      const payload = source === "network"
        ? networkPayload()
        : {
            incident_id: form.incident_id,
            user_id: form.user_id,
            start_hour: Number(form.start_hour),
            end_hour: Number(form.end_hour),
            limit: Number(form.limit),
          };
      const searchResult = await postJson(
        source === "network" ? "/api/search/network" : "/api/search/lanl",
        payload,
      );
      setResult(searchResult);
      setDecisions({});
      const audit = await getJson(
        `/api/responses/${encodeURIComponent(form.incident_id)}`,
      );
      setDecisionHistory(audit.decisions);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function generateReport() {
    setBusy(true);
    setError("");
    try {
      const body = await postJson("/api/report/network", networkPayload());
      setResult(body);
      setReport(body.report);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function decide(proposal, decision) {
    setBusy(true);
    setError("");
    try {
      const outcome = await postJson("/api/responses/decision", {
        proposal_id: proposal.proposal_id,
        decision,
        analyst,
        reason,
      });
      setDecisions((current) => ({
        ...current,
        [proposal.proposal_id]: outcome,
      }));
      const audit = await getJson(
        `/api/responses/${encodeURIComponent(proposal.incident_id)}`,
      );
      setDecisionHistory(audit.decisions);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function searchKnowledge() {
    setBusy(true);
    setError("");
    try {
      const result = await postJson("/api/knowledge/search", {
        query: source === "network"
          ? `${form.entity} network discovery port fan-out Sigma`
          : `${form.user_id} authentication LANL ATT&CK Sigma`,
        limit: 5,
      });
      setResult((current) => ({ ...current, knowledge_references: result.references }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-mark">S</span>
          <span>SOC <b>ANALYST</b></span>
        </a>
        <div className="mode-pill"><i /> LOCAL INVESTIGATION · DRY RUN</div>
      </header>

      <section className="page-heading">
        <div>
          <p className="eyebrow">SECURITY OPERATIONS CENTER</p>
          <h1>Incident investigation</h1>
          <p className="subtitle">Search evidence, review hypotheses, and document decisions.</p>
        </div>
        <button className="button button-muted" onClick={search} disabled={busy}>
          {busy ? "Working…" : "Refresh evidence"}
        </button>
      </section>

      <section className="workspace">
        <aside className="panel search-panel">
          <div className="panel-title">
            <div><p className="eyebrow">INVESTIGATION</p><h2>Search parameters</h2></div>
          </div>
          <label>Evidence source
            <select value={source} onChange={(event) => {
              setSource(event.target.value);
              setResult(null);
              setReport(null);
            }}>
              <option value="network">CICIDS2017 network flows</option>
              <option value="lanl">LANL login user-hours</option>
            </select>
          </label>
          <label>Incident ID
            <input value={form.incident_id} onChange={(event) => update("incident_id", event.target.value)} />
          </label>
          {source === "network" ? (
            <>
              <label>Entity IP
                <input value={form.entity} onChange={(event) => update("entity", event.target.value)} />
              </label>
              <div className="field-row">
                <label>Start time<input type="datetime-local" value={form.start_time} onChange={(event) => update("start_time", event.target.value)} /></label>
                <label>End time<input type="datetime-local" value={form.end_time} onChange={(event) => update("end_time", event.target.value)} /></label>
              </div>
              <details>
                <summary>Optional exact filters</summary>
                <div className="filter-stack">
                  <label>Source IP<input value={form.src_ip} onChange={(event) => update("src_ip", event.target.value)} /></label>
                  <label>Destination IP<input value={form.dst_ip} onChange={(event) => update("dst_ip", event.target.value)} /></label>
                  <div className="field-row">
                    <label>Source port<input type="number" min="0" max="65535" value={form.source_port} onChange={(event) => update("source_port", event.target.value)} /></label>
                    <label>Destination port<input type="number" min="0" max="65535" value={form.destination_port} onChange={(event) => update("destination_port", event.target.value)} /></label>
                  </div>
                  <label>Protocol<input placeholder="tcp / udp / icmp" value={form.protocol} onChange={(event) => update("protocol", event.target.value)} /></label>
                </div>
              </details>
            </>
          ) : (
            <>
              <label>LANL user ID
                <input value={form.user_id} onChange={(event) => update("user_id", event.target.value)} />
              </label>
              <p className="field-note">LANL time is relative hour index, not wall-clock time.</p>
              <div className="field-row">
                <label>Start hour<input type="number" min="0" value={form.start_hour} onChange={(event) => update("start_hour", event.target.value)} /></label>
                <label>End hour<input type="number" min="0" value={form.end_hour} onChange={(event) => update("end_hour", event.target.value)} /></label>
              </div>
            </>
          )}
          <label>Maximum events
            <input type="number" min="1" max="500" value={form.limit} onChange={(event) => update("limit", event.target.value)} />
          </label>
          <button className="button button-primary full-width" onClick={search} disabled={busy}>
            Search read-only evidence
          </button>
          {source === "network" && (
            <button className="button button-outline full-width" onClick={generateReport} disabled={busy || !events.length}>
              Generate Groq investigation report
            </button>
          )}
          <p className="field-note">Search is read-only. Report generation sends selected evidence to Groq.</p>
        </aside>

        <section className="results-column">
          {error && <div className="alert alert-error" role="alert">{error}</div>}
          {result ? (
            <>
              <section className="stats-row">
                <div className="stat-card"><span>Evidence records</span><strong>{result.evidence_count}</strong></div>
                <div className="stat-card"><span>Source</span><strong className="stat-source">{result.source}</strong></div>
                <div className="stat-card"><span>Completeness</span><strong className={result.possibly_truncated ? "warning-text" : "good-text"}>{result.possibly_truncated ? "Possibly truncated" : "Within limit"}</strong></div>
              </section>
              {result.possibly_truncated && <div className="alert alert-warning">Results reached the configured limit. ATT&amp;CK hypotheses are withheld because the search may be incomplete.</div>}

              <section className="panel">
                <div className="panel-title"><div><p className="eyebrow">EVENTS</p><h2>Incident timeline</h2></div><span className="count-badge">{timeline.length} records</span></div>
                {timeline.length ? (
                  <div className="timeline">
                    {timeline.map((event) => (
                      <article className="timeline-event" key={event.evidence_id}>
                        <div className="timeline-marker" />
                        <div className="event-card">
                          <div className="event-header"><time>{formatTime(event, result.source)}</time><code>{event.evidence_id}</code></div>
                          <EventFacts event={event} source={result.source} />
                        </div>
                      </article>
                    ))}
                  </div>
                ) : <p className="empty-state">No records matched this search. This is not proof that no activity occurred.</p>}
              </section>

              {result.attack_mapping && (
                <section className="panel">
                  <div className="panel-title"><div><p className="eyebrow">BEHAVIORAL CONTEXT</p><h2>ATT&amp;CK hypotheses</h2></div><span className="status-tag">{result.attack_mapping.status.replaceAll("_", " ")}</span></div>
                  {result.attack_mapping.hypotheses?.length ? result.attack_mapping.hypotheses.map((item) => (
                    <article className="hypothesis-card" key={`${item.technique_id}:${item.observed_target}`}>
                      <div><span className="technique-id">{item.technique_id}</span><b>{item.technique_name}</b><span className="confidence">{item.confidence} confidence</span></div>
                      <p>{item.assessment}</p>
                      <div className="reference-list">{item.evidence_ids.map((id) => <code key={id}>{id}</code>)}</div>
                      <a href={item.reference} target="_blank" rel="noreferrer">MITRE ATT&amp;CK reference ↗</a>
                    </article>
                  )) : <p className="field-note">{result.attack_mapping.notice || "No mapping candidates."}</p>}
                </section>
              )}

              {result.knowledge_references && (
                <section className="panel">
                  <div className="panel-title">
                    <div><p className="eyebrow">LOCAL KNOWLEDGE</p><h2>ATT&amp;CK and Sigma references</h2></div>
                    <button className="button button-small button-muted" onClick={searchKnowledge} disabled={busy}>Retrieve</button>
                  </div>
                  <div className="knowledge-grid">
                    {result.knowledge_references.map((reference) => (
                      <article className="knowledge-card" key={reference.reference_id}>
                        <span className="reference-kind">{reference.kind}</span>
                        <b>{reference.title}</b>
                        <p>{reference.text}</p>
                        <a href={reference.source_url.startsWith("http") ? reference.source_url : "#"} target="_blank" rel="noreferrer">{reference.citation} ↗</a>
                      </article>
                    ))}
                  </div>
                  <p className="field-note">Reference knowledge is separate from observed incident evidence.</p>
                </section>
              )}

              {report && (
                <section className="panel report-panel">
                  <div className="panel-title"><div><p className="eyebrow">LLM-ASSISTED</p><h2>Investigation report</h2></div><span className="status-tag">{report.verification?.verified ? "verified by model" : "not verified"}</span></div>
                  <p>{report.summary}</p>
                  <div className="reference-list">{report.summary_evidence_ids?.map((id) => <code key={id}>{id}</code>)}</div>
                  {report.summary_reference_ids?.length > 0 && <div className="reference-list">{report.summary_reference_ids.map((id) => <code key={id}>{id}</code>)}</div>}
                  {report.claims?.map((claim, index) => (
                    <article className="claim-card" key={index}><p>{claim.statement}</p><div className="reference-list">{claim.evidence_ids.map((id) => <code key={id}>{id}</code>)}{claim.reference_ids?.map((id) => <code key={id}>{id}</code>)}</div></article>
                  ))}
                  <p className="field-note">Model verification is a safeguard, not a formal proof. Analyst review is required.</p>
                </section>
              )}

              {result.response_proposals?.length > 0 && (
                <section className="panel">
                  <div className="panel-title"><div><p className="eyebrow">HUMAN REVIEW REQUIRED</p><h2>Dry-run response drafts</h2></div><span className="dry-tag">NO ACTION EXECUTES</span></div>
                  <div className="review-fields">
                    <label>Analyst<input value={analyst} onChange={(event) => setAnalyst(event.target.value)} placeholder="Your name or analyst ID" /></label>
                    <label>Decision reason<input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why approve or reject this draft?" /></label>
                  </div>
                  {result.response_proposals.map((proposal) => (
                    <article className="proposal-card" key={proposal.proposal_id}>
                      <div><b>{proposal.action.replaceAll("_", " ")}</b><span className="status-tag">{decisions[proposal.proposal_id]?.decision || proposal.status}</span></div>
                      <p>{proposal.rationale}</p>
                      <div className="reference-list">{proposal.evidence_ids.map((id) => <code key={id}>{id}</code>)}</div>
                      {!decisions[proposal.proposal_id] && (
                        <div className="button-row">
                          <button className="button button-primary button-small" disabled={busy || !analyst.trim() || !reason.trim()} onClick={() => decide(proposal, "approve")}>Approve dry-run</button>
                          <button className="button button-muted button-small" disabled={busy || !analyst.trim() || !reason.trim()} onClick={() => decide(proposal, "reject")}>Reject</button>
                        </div>
                      )}
                      {decisions[proposal.proposal_id] && <p className="field-note">{decisions[proposal.proposal_id].notice}</p>}
                    </article>
                  ))}
                </section>
              )}
              {decisionHistory.length > 0 && (
                <section className="panel">
                  <div className="panel-title"><div><p className="eyebrow">AUDIT TRAIL</p><h2>Recorded dry-run decisions</h2></div><span className="count-badge">{decisionHistory.length} decisions</span></div>
                  {decisionHistory.map((item) => (
                    <article className="claim-card" key={item.decision_id}>
                      <div className="event-header"><b>{item.decision} · {item.analyst}</b><time>{new Date(item.decided_at).toLocaleString()}</time></div>
                      <p>{item.reason}</p>
                      <span className="dry-tag">{item.execution_mode} · executed: {String(item.executed)}</span>
                    </article>
                  ))}
                </section>
              )}
            </>
          ) : (
            <section className="empty-workspace">
              <div className="empty-icon">⌕</div>
              <h2>Start an investigation</h2>
              <p>Choose a data source, set the scope, and run a read-only evidence search.</p>
              <span>Available sources: CICIDS2017 network flows and processed LANL user-hour features.</span>
            </section>
          )}
        </section>
      </section>
      <footer>Local dashboard · Read-only telemetry · Human approval required · Dry run only</footer>
    </main>
  );
}

export default App;
