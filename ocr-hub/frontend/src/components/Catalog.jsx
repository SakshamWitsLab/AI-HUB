import React, { useMemo, useState } from 'react'
import { api } from '../api'
import ModelDetail from './ModelDetail'

function ModelCard({ m, onDownload, reload, dlJob }) {
  const [busy, setBusy] = useState(false)
  const [detail, setDetail] = useState(false)
  const installed = m.installed && !m.hf_id.startsWith('builtin:')
  const builtin = m.hf_id.startsWith('builtin:')
  const status = dlJob?.status
  const dlActive = ['running', 'queued', 'pausing'].includes(status)
  const dlPaused = status === 'paused'
  const dlErr = status === 'error' ? dlJob?.error : null
  const pct = dlJob?.progress ?? 0

  const uninstall = async () => {
    setBusy(true)
    try {
      await fetch(`/api/models/${encodeURIComponent(m.hf_id)}`, { method: 'DELETE' })
      reload()
    } catch (e) { alert(e.message) }
    setBusy(false)
  }

  return (
    <article className="mcard flex flex-col">
      <div className="flex items-center gap-2">
        <span className="eng">{m.engine}</span>
        {m.tier && <span className={`tag ${m.tier === 'user' ? 'user' : m.tier === 'starter' ? 'starter' : ''}`}>{m.tier}</span>}
      </div>
      <h3 className="mt-2">{m.name}</h3>
      <code className="hf">{m.hf_id}</code>
      <p className="notes">{m.notes}</p>
      <div className="meta">
        <span>{m.size_gb} GB</span>
        <span>{m.params}</span>
        <span>{m.languages}</span>
        <span>{m.task}</span>
        <span title="popularity on HuggingFace">⬇ {(m.downloads || 0).toLocaleString()}</span>
      </div>

      <div className="mt-3 flex items-center gap-2" style={{ borderTop: '1px dashed rgba(128,128,128,.3)', paddingTop: 10 }}>
        <button className="btn ghost !py-1 !px-2 text-xs" onClick={() => setDetail(true)}>
          ⚙ Details
        </button>
        <a className="btn ghost !py-1 !px-2 text-xs" href={`https://huggingface.co/${m.hf_id}`}
          target="_blank" rel="noreferrer" title="Open the official model card on HuggingFace"
          style={{ opacity: m.hf_id.startsWith('builtin:') || m.cloud ? 0.35 : 1, pointerEvents: m.hf_id.startsWith('builtin:') || m.cloud ? 'none' : 'auto' }}>
          Open on HF ↗
        </a>
        <div className="ml-auto flex items-center gap-2" style={{ whiteSpace: 'nowrap' }}>
          {dlErr ? (
            <span className="text-[12px]" style={{ color: '#b42318' }} title={dlErr}>Download failed</span>
          ) : m.cloud ? (
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
        ) : dlActive ? (
          <button className="btn ghost !py-1 !px-2 text-xs" disabled={busy} onClick={async () => {
            setBusy(true); try { await api.pauseDownload(m.hf_id) } catch (e) { alert(e.message) }; setBusy(false)
          }}>⏸ Pause</button>
        ) : dlPaused ? (
          <button className="btn ghost !py-1 !px-2 text-xs" disabled={busy} onClick={async () => {
            setBusy(true); try { await api.download(m.hf_id) } catch (e) { alert(e.message) }; setBusy(false)
          }}>▶ Resume</button>
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
      </div>

      {(dlActive || dlPaused || dlErr) && (
        <div className="mt-2">
          {dlErr ? (
            <div className="err text-[12px]">{dlErr}</div>
          ) : (
            <>
              <div style={{ height: 6, borderRadius: 999, background: 'color-mix(in srgb, var(--ink-soft) 20%, transparent)', overflow: 'hidden' }}>
                <div style={{
                  width: `${pct}%`, height: '100%', borderRadius: 999,
                  background: dlPaused ? 'var(--amber)' : 'var(--scan)',
                  transition: 'width .6s ease',
                }} />
              </div>
              <div className="flex justify-between text-[11px] mono mt-1" style={{ color: 'var(--ink-soft)' }}>
                <span>{dlPaused ? 'Paused' : 'Downloading'} · {pct}%</span>
                <span>{m.name}</span>
              </div>
            </>
          )}
        </div>
      )}

      {detail && (
        <ModelDetail m={m} onClose={() => setDetail(false)}
          onRun={(hfId) => { location.hash = `#playground:${hfId}`; onClose?.() }} />
      )}
    </article>
  )
}

export default function Catalog({ models, reload, dlJobs = {} }) {
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
            {filt(starter).map((m) => <ModelCard key={m.hf_id} m={m} reload={reload} dlJob={dlJobs[m.hf_id]} />)}
          </div>
        </>
      )}
      {filt(starter).length === 0 && filter !== 'all' && (
        <p className="text-sm mb-6" style={{ color: 'var(--ink-soft)' }}>No starter models match “{filter}”.</p>
      )}

      <h3 className="mb-3 text-lg font-semibold tracking-tight">Model library</h3>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {filt(lib).map((m) => <ModelCard key={m.hf_id} m={m} reload={reload} dlJob={dlJobs[m.hf_id]} />)}
        {filt(lib).length === 0 && (
          <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>Nothing matches “{filter}” here.</p>
        )}
      </div>
    </>
  )
}
