import React, { useMemo, useState } from 'react'
import { api } from '../api'

function ModelCard({ m, onDownload, reload }) {
  const [busy, setBusy] = useState(false)
  const installed = m.installed && !m.hf_id.startsWith('builtin:')
  const builtin = m.hf_id.startsWith('builtin:')

  const uninstall = async () => {
    setBusy(true)
    try {
      await fetch(`/api/models/${encodeURIComponent(m.hf_id)}`, { method: 'DELETE' })
      reload()
    } catch (e) { alert(e.message) }
    setBusy(false)
  }

  return (
    <article className="mcard">
      <span className="eng">{m.engine}</span>
      {m.tier && <span className={`tag ${m.tier === 'user' ? 'user' : m.tier === 'starter' ? 'starter' : ''}`}>{m.tier}</span>}
      <h3>{m.name}</h3>
      <code className="hf">{m.hf_id}</code>
      <p className="notes">{m.notes}</p>
      <div className="meta">
        <span>{m.size_gb} GB</span>
        <span>{m.params}</span>
        <span>{m.languages}</span>
        <span>{m.task}</span>
        <span title="popularity on HuggingFace">⬇ {(m.downloads || 0).toLocaleString()}</span>
      </div>
      <div className="mt-1 flex items-center gap-2">
        {m.cloud ? (
          <span className="text-[12px]" style={{ color: 'var(--amber)' }}>
            Cloud — needs the OXR switch in Settings
          </span>
        ) : installed ? (
          <>
            <span style={{ color: 'var(--scan)' }} className="text-sm font-medium">Installed</span>
            <button className="btn ghost !py-1 !px-2 text-xs ml-auto" disabled={busy} onClick={uninstall}>
              {busy ? 'Removing…' : 'Uninstall'}
            </button>
          </>
        ) : builtin ? (
          <span className="text-[12px]" style={{ color: 'var(--ink-soft)' }}>Part of this hub — nothing to download</span>
        ) : (
          <button className="btn primary" disabled={busy || m.downloading} onClick={async () => {
            setBusy(true)
            try { await api.download(m.hf_id) } catch (e) { alert(e.message) }
            setBusy(false)
          }}>
            {m.downloading ? 'Downloading…' : 'Download model'}
          </button>
        )}
      </div>
    </article>
  )
}

export default function Catalog({ models, reload }) {
  const [hfRepo, setHfRepo] = useState('')
  const [preview, setPreview] = useState(null)
  const [err, setErr] = useState('')
  const [filter, setFilter] = useState('all')

  const check = async () => {
    setErr(''); setPreview(null)
    try {
      const meta = await api.preview(hfRepo.trim())
      setPreview(meta)
    } catch (e) { setErr(e.message) }
  }

  const starter = models.filter((m) => m.tier === 'starter')
  const lib = models.filter((m) => m.tier !== 'starter')

  const filt = (list) =>
    list.filter((m) =>
      filter === 'all' ? true
      : filter === 'under1' ? m.size_gb < 1
      : filter === 'installed' ? (m.installed && !m.hf_id.startsWith('builtin:'))
      : true,
    )

  return (
    <>
      <div className="page-head">
        <h2>Library</h2>
        <div className="flex gap-2">
          {[['all', 'All'], ['under1', 'Under 1 GB'], ['installed', 'Installed']].map(([k, l]) => (
            <button key={k} className={`chip ${filter === k ? 'on' : ''}`} onClick={() => setFilter(k)}>{l}</button>
          ))}
        </div>
      </div>
      <p className="page-sub mb-6">
        Cards live here before you install them. Pick a starter model to try, or pull anything
        from HuggingFace by its id.
      </p>

      {/* Fetch from HuggingFace */}
      <section className="mcard mb-8 !flex-row !items-center gap-3">
        <input
          className="input mono flex-1"
          placeholder="huggingface id — e.g. PaddlePaddle/PaddleOCR-VL-1.6"
          value={hfRepo}
          onChange={(e) => setHfRepo(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && check()}
        />
        <button className="btn ghost" onClick={check}>Inspect</button>
        {preview && (
          <button className="btn primary" onClick={async () => {
            await api.add(preview.hf_id)
            try { await api.download(preview.hf_id) } catch {}
            setPreview(null); setHfRepo(''); reload()
          }}>
            Add & download ({preview.size_gb} GB)
          </button>
        )}
        {err && <span className="err">{err}</span>}
        {preview && !err && (
          <span className="text-xs" style={{ color: 'var(--ink-soft)' }}>
            {preview.name} — {preview.license}, ⬇{preview.downloads.toLocaleString()}{preview.gated ? ', needs token' : ''}
          </span>
        )}
      </section>

      {starter.length > 0 && (
        <>
          <h3 className="mb-3 text-lg font-semibold tracking-tight">Ready in seconds — under 1 GB</h3>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3 mb-8">
            {filt(starter).map((m) => <ModelCard key={m.hf_id} m={m} reload={reload} />)}
          </div>
        </>
      )}
      {filt(starter).length === 0 && filter !== 'all' && (
        <p className="text-sm mb-6" style={{ color: 'var(--ink-soft)' }}>No starter models match “{filter}”.</p>
      )}

      <h3 className="mb-3 text-lg font-semibold tracking-tight">Model library</h3>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {filt(lib).map((m) => <ModelCard key={m.hf_id} m={m} reload={reload} />)}
        {filt(lib).length === 0 && (
          <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>Nothing matches “{filter}” here.</p>
        )}
      </div>
    </>
  )
}
