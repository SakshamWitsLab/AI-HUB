import React, { useState } from 'react'
import { api } from '../api'

const UNKNOWN = 'unknown'

/* rows: [label, model => value, model => ?? shorter|higher] */
const ROWS = [
  ['Type', (m) => m.facts.kind, null],
  ['What it handles', (m) => m.facts.scope, null],
  ['Output', (m) => m.facts.output, null],
  ['Hardware needs', (m) => m.facts.hardware, null],
  ['Standout', (m) => m.facts.strengths, null],
  ['Size on disk', (m) => m.live.size_gb != null && m.live.size_gb !== 0 ? `${m.live.size_gb} GB` : null, 'lower'],
  ['Parameters', (m) => m.live.params, null],
  ['Languages', (m) => m.card.languages, null],
  ['License', (m) => m.live.license, null],
  ['HF downloads', (m) => (m.live.downloads ? m.live.downloads.toLocaleString() : null), 'higher'],
  ['HF likes', (m) => (m.live.likes ? m.live.likes.toLocaleString() : null), 'higher'],
  ['Status', (m) => m.card.installed ? 'Installed'
      : m.card.cloud ? 'Cloud (Settings switch)'
      : m.live.gated ? 'On HF — gated'
      : 'On HuggingFace', null],
]

const Cell = ({ v, best }) =>
  v == null || v === '' ? (
    <span style={{ color: 'var(--ink-soft)', fontStyle: 'italic' }}>{UNKNOWN} — not verified</span>
  ) : (
    <span style={best ? { fontWeight: 600, color: 'var(--scan)' } : undefined}>{v}</span>
  )

export default function Compare({ models, initialSelected }) {
  const [selected, setSelected] = useState(initialSelected || [])
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  const toggle = (id) =>
    setSelected((s) => s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-4))

  const runCompare = async (ids = selected) => {
    if (!ids.length) { setErr('Pick at least one model'); return }
    setLoading(true); setErr('')
    try { setData(await api.compare(ids)) }
    catch (e) { setErr(e.message) }
    setLoading(false)
  }

  // highlight the "best" cell in a row for counter-style metrics
  const withBest = (rows, models) => {
    if (!models) return {}
    const best = {}
    for (const [label, get, who] of rows) {
      if (!who) continue
      const vals = models
        .map((m) => (get(m) ?? '').toString())
        .map((s) => parseFloat(s.replace(/[^0-9.]/g, '')))
        .map((n) => (Number.isFinite(n) ? n : who === 'lower' ? Infinity : -Infinity))
      const picks = who === 'lower' ? Math.min(...vals) : Math.max(...vals)
      models.forEach((m, i) => { if (vals[i] === picks && picks !== (who === 'lower' ? Infinity : -Infinity) && models.length > 1) best[`${label}:${m.card.hf_id}`] = true })
    }
    return best
  }

  const bestMap = data && withBest(ROWS, data.models)

  return (
    <>
      <div className="page-head"><h2>Compare</h2></div>
      <p className="page-sub mb-4">
        Side-by-side specs for catalog models. Specs marked with a date are fetched live from
        HuggingFace at compare time; the rest comes from each model's official card. Fields we
        can't verify show “unknown — not verified”, never a guess.
      </p>

      <div className="mcard mb-6 flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <b className="text-sm">Models</b>
          {models.map((m) => (
            <button key={m.hf_id} className={`chip ${selected.includes(m.hf_id) ? 'on' : ''}`}
              onClick={() => toggle(m.hf_id)}>
              {m.name}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button className="btn primary" disabled={loading || !selected.length}
            onClick={() => runCompare()}>
            {loading ? 'Fetching…' : `Compare ${selected.length} model${selected.length === 1 ? '' : 's'}`}
          </button>
          {selected.length > 0 && (
            <button className="btn ghost" onClick={() => { setSelected([]); setData(null) }}>Clear</button>
          )}
          {err && <span className="err">{err}</span>}
        </div>
      </div>

      {data && data.models.length > 0 && (
        <div className="mcard mb-6" style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 640 }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--ink-soft)' }}>Spec</th>
                {data.models.map((m) => (
                  <th key={m.card.hf_id} style={{ textAlign: 'left', padding: '8px 12px' }}>
                    <div style={{ fontWeight: 700 }}>{m.card.name}</div>
                    <code className="hf" style={{ fontSize: 11 }}>{m.card.hf_id}</code>
                    <div className="mt-1">
                      {m.live.ok && m.asof && (
                        <span className="text-[10px] mono" style={{ color: 'var(--ink-soft)' }}>
                          HuggingFace · {m.asof}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map(([label, get, who]) => (
                <tr key={label} style={{ borderTop: '1px solid var(--line, rgba(128,128,128,.2))' }}>
                  <td style={{ padding: '8px 12px', color: 'var(--ink-soft)', whiteSpace: 'nowrap', verticalAlign: 'top' }}>{label}</td>
                  {data.models.map((m) => (
                    <td key={m.card.hf_id} style={{ padding: '8px 12px', verticalAlign: 'top' }}>
                      <Cell v={get(m)} best={bestMap[`${label}:${m.card.hf_id}`]} />
                    </td>
                  ))}
                </tr>
              ))}
              <tr style={{ borderTop: '1px solid var(--line, rgba(128,128,128,.2))' }}>
                <td style={{ padding: '8px 12px', color: 'var(--ink-soft)' }}>Links</td>
                {data.models.map((m) => (
                  <td key={m.card.hf_id} style={{ padding: '8px 12px' }}>
                    {m.live.url ? (
                      <a href={m.live.url} target="_blank" rel="noreferrer">Model card on HuggingFace ↗</a>
                    ) : (
                      <span style={{ color: 'var(--ink-soft)', fontStyle: 'italic' }}>
                        built-in engine — ships with the hub
                      </span>
                    )}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {!data && !loading && (
        <div className="mcard text-center text-sm" style={{ color: 'var(--ink-soft)' }}>
          Pick 2–4 models and hit Compare. For a real head-to-head on your own file, run both
          models in the Playground and compare the output cards side by side.
        </div>
      )}
    </>
  )
}
