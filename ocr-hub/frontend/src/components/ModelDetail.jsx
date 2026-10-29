import React, { useEffect, useState } from 'react'
import { api } from '../api'

/* Full-screen model detail view — "HF viewer"-style immersive page.
   Every architecture number comes verbatim from the model authors'
   published config.json on HuggingFace. Nothing is invented. */

const num = (v) => (v == null ? null : (typeof v === 'number' ? v.toLocaleString() : v))

function Stack({ section }) {
  const flow = [
    section.max_position_embeddings && {
      t: 'Embeddings',
      d: section.vocab_size ? `vocab ${num(section.vocab_size)}` : '',
    },
    section.num_hidden_layers && {
      t: `Transformer × ${section.num_hidden_layers}`,
      d: [
        section.num_attention_heads && `${section.num_attention_heads} heads`,
        section.num_key_value_heads && `${section.num_key_value_heads} KV`,
        section.hidden_size && `hidden ${num(section.hidden_size)}`,
        section.intermediate_size && `MLP ${num(section.intermediate_size)}`,
      ].filter(Boolean).join(' · '),
    },
    section.max_position_embeddings && {
      t: 'LM head',
      d: section.vocab_size ? `vocab ${num(section.vocab_size)}` : '',
    },
  ].filter(Boolean)

  return (
    <div className="glass-block arch-stack rounded-xl" style={{ padding: '14px 16px' }}>
      <div className="flex items-baseline justify-between gap-2">
        <b style={{ fontSize: 13.5 }}>{section.label}</b>
        {section.arch && <code className="hf" style={{ fontSize: 10.5 }}>{section.arch}</code>}
      </div>
      <div className="text-[11px] mono" style={{ color: 'var(--ink-soft)' }}>
        type: {section.model_type || '—'}{section.dtype ? ` · ${section.dtype}` : ''}
      </div>
      <div className="mt-2">
        {flow.map((b, i) => (
          <React.Fragment key={i}>
            {i > 0 && <div className="arch-arrow" aria-hidden>▼</div>}
            <div className="flow-block" style={{ padding: '8px 11px' }}>
              <div style={{ fontWeight: 600 }}>{b.t}</div>
              <div className="text-[11px]" style={{ color: 'var(--ink-soft)' }}>{b.d}</div>
            </div>
          </React.Fragment>
        ))}
      </div>
      {section.max_position_embeddings && (
        <div className="text-[10.5px] mt-2" style={{ color: 'var(--ink-soft)' }}>
          context {num(section.max_position_embeddings)} tokens
          {section.hidden_act ? ` · activation ${section.hidden_act}` : ''}
        </div>
      )}
    </div>
  )
}

const Section = ({ title, children }) => (
  <section className="mb-6">
    <h3 className="text-xs font-semibold tracking-wide uppercase mb-2" style={{ color: 'var(--ink-soft)' }}>{title}</h3>
    {children}
  </section>
)

export default function ModelDetail({ m, onClose, onRun }) {
  const [arch, setArch] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [showCfg, setShowCfg] = useState(false)
  const builtin = m.hf_id?.startsWith('builtin:')

  useEffect(() => {
    let live = true
    setLoading(true)
    api.arch(m.hf_id)
      .then((d) => live && setArch(d))
      .catch((e) => live && setArch({ available: false, reason: e.message }))
      .finally(() => live && setLoading(false))
    const esc = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    document.body.style.overflow = 'hidden'
    return () => { live = false; window.removeEventListener('keydown', esc); document.body.style.overflow = '' }
  }, [m.hf_id])

  const doDownload = async () => {
    setBusy(true)
    try { await api.download(m.hf_id) } catch (e) { window.appToast?.(`!${e.message}`, 'err') }
    setBusy(false)
  }
  const doUninstall = async () => {
    setBusy(true)
    try {
      await fetch(`/api/models/${encodeURIComponent(m.hf_id)}`, { method: 'DELETE' })
      window.appToast?.(`${m.name} removed`)
    } catch (e) { window.appToast?.(`!${e.message}`, 'err') }
    setBusy(false)
  }

  const stackData = arch?.available
    ? (arch.subs?.length
        ? [...arch.subs.map((s) => ({ ...s, label: s.name })), { ...arch.main, label: 'Main model' }]
        : [{ ...arch.main, label: 'Architecture' }])
    : []
  const facts = arch?.facts || {}
  const live = arch?.live

  const specRows = [
    ['Type', facts.kind || m.task],
    ['Handles', facts.scope],
    ['Output', facts.output],
    ['Hardware', facts.hardware],
    ['Languages', m.languages],
    ['Size on disk', m.size_gb ? `${m.size_gb} GB` : null],
    ['Parameters', m.params],
    ['License', live?.license],
    ['HF popularity', live ? `⬇ ${num(live.downloads)} · ♥ ${num(live.likes)}` : null],
  ].filter(([, v]) => v != null)

  return (
    <div className="fixed inset-0 z-50 flex" role="dialog" aria-modal="true"
      style={{ background: 'rgba(10,12,18,.6)', backdropFilter: 'blur(6px)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="w-full h-full overflow-y-auto"
        style={{ background: 'var(--paper)', color: 'var(--ink)' }}>
        <div className="max-w-4xl mx-auto px-5 py-6">
          {/* header */}
          <div className="flex items-start gap-3">
            <div className="flex-1 min-w-0">
              <h2 className="text-2xl font-bold tracking-tight">{m.name}</h2>
              <code className="hf text-[12px]" style={{ overflowWrap: 'anywhere' }}>{m.hf_id}</code>
              <div className="flex flex-wrap items-center gap-2 mt-2">
                <span className="eng">{m.engine}</span>
                {m.tier && <span className={`tag ${m.tier === 'user' ? 'user' : m.tier === 'starter' ? 'starter' : ''}`}>{m.tier}</span>}
                {m.cloud && <span className="tag">cloud</span>}
                {m.installed && !builtin && <span className="tag" style={{ color: 'var(--scan)' }}>installed</span>}
                {live?.asof && <span className="text-[11px] mono" style={{ color: 'var(--ink-soft)' }}>HuggingFace · {live.asof}</span>}
              </div>
            </div>
            <button className="btn ghost" onClick={onClose} title="Close (Esc)">✕</button>
          </div>

          {/* action bar */}
          <div className="flex flex-wrap items-center gap-2 my-5">
            {m.cloud ? (
              <span className="text-[12px]" style={{ color: 'var(--amber)' }}>Cloud model — needs the OXR switch in Settings</span>
            ) : m.installed && !builtin ? (
              <>
                <span className="text-sm font-medium" style={{ color: 'var(--scan)' }}>Installed</span>
                <button className="btn ghost !py-1 !px-2 text-xs" disabled={busy} onClick={doUninstall}>
                  {busy ? 'Removing…' : 'Uninstall'}
                </button>
              </>
            ) : builtin ? (
              <span className="text-[12px]" style={{ color: 'var(--ink-soft)' }}>Part of this hub — nothing to download</span>
            ) : (
              <button className="btn primary !py-1 !px-3 text-xs" disabled={busy || m.downloading} onClick={doDownload}>
                {m.downloading ? 'Downloading…' : 'Download model'}
              </button>
            )}
            {!builtin && !m.cloud && (
              <a className="btn ghost !py-1 !px-2 text-xs" href={`https://huggingface.co/${m.hf_id}`} target="_blank" rel="noreferrer">
                Open on HF ↗
              </a>
            )}
            {onRun && (m.installed || builtin || m.cloud) && (
              <button className="btn ghost !py-1 !px-2 text-xs" onClick={() => onRun(m.hf_id)}>Run in Playground →</button>
            )}
          </div>

          {/* about */}
          <Section title="About this model">
            <div className="glass-block rounded-xl" style={{ padding: '14px 16px' }}>
              <div className="grid gap-x-8 gap-y-2" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
                {specRows.map(([k, v]) => (
                  <div key={k} className="flex gap-2 text-[13px]">
                    <span style={{ color: 'var(--ink-soft)', minWidth: 108, flexShrink: 0 }}>{k}</span>
                    <span style={{ overflowWrap: 'anywhere' }}>{String(v)}</span>
                  </div>
                ))}
              </div>
              <p className="text-[13px] mt-3" style={{ color: 'var(--ink-soft)' }}>{m.notes}</p>
            </div>
          </Section>

          {/* architecture */}
          <Section title="Architecture">
            {loading && <>
              <div className="shimmer" style={{ height: 60 }} />
              <div className="shimmer mt-2" style={{ height: 140 }} />
            </>}
            {!loading && arch && !arch.available && (
              <div className="glass-block rounded-xl text-sm" style={{ padding: '14px 16px', color: 'var(--ink-soft)' }}>
                No architecture config available — {arch.reason}
              </div>
            )}
            {!loading && arch?.available && (
              <>
                <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))' }}>
                  {stackData.map((s, i) => <Stack key={i} section={s} />)}
                </div>
                <div className="mt-3 text-xs">
                  <a href={arch.source_url} target="_blank" rel="noreferrer">view source → config.json ↗</a>
                  {!builtin && !m.cloud && (
                    <> · <a href={`https://huggingface.co/${m.hf_id}`} target="_blank" rel="noreferrer">official model card ↗</a></>
                  )}
                </div>
                {showCfg && (
                  <div className="rounded-xl mt-3" style={{ border: '1px solid var(--line)', background: 'var(--panel)' }}>
                    <div className="flex items-center justify-between px-3 py-2" style={{ borderBottom: '1px solid var(--line)' }}>
                      <b className="text-xs">config.json — as published on HuggingFace</b>
                      <button className="btn ghost !py-0.5 !px-2 text-xs"
                        onClick={() => navigator.clipboard.writeText(JSON.stringify(arch.raw, null, 2))}>
                        copy
                      </button>
                    </div>
                    <pre className="text-[11px] mono m-3 p-3 rounded-lg overflow-auto"
                      style={{ background: 'color-mix(in srgb, var(--ink) 5%, var(--panel))', maxHeight: 340 }}>
                      {JSON.stringify(arch.raw, null, 2)}
                    </pre>
                  </div>
                )}
                <div className="mt-2 text-xs">
                  <button className="btn ghost !py-0.5 !px-2 text-xs" onClick={() => setShowCfg((s) => !s)}>
                    {showCfg ? '▾ Hide raw config' : '⚙ HF config viewer'}
                  </button>
                </div>
              </>
            )}
          </Section>

          {/* hf metadata */}
          <Section title="HuggingFace listing">
            <div className="glass-block rounded-xl text-[13px]" style={{ padding: '14px 16px' }}>
              {builtin || m.cloud ? (
                <span style={{ color: 'var(--ink-soft)' }}>Built into this hub or served by its provider — no public HuggingFace listing.</span>
              ) : live ? (
                <>
                  <div className="flex flex-wrap gap-x-6 gap-y-1">
                    <span>⬇ {num(live.downloads)} downloads</span>
                    <span>♥ {num(live.likes)} likes</span>
                    <span>License: {live.license}</span>
                    {live.gated && <span style={{ color: 'var(--amber)' }}>gated repo</span>}
                    <span className="mono text-[11px]" style={{ color: 'var(--ink-soft)' }}>as of {live.asof}</span>
                  </div>
                  <div className="mt-2"><a href={live.url} target="_blank" rel="noreferrer">huggingface.co/{m.hf_id} ↗</a></div>
                </>
              ) : (
                <span style={{ color: 'var(--ink-soft)' }}>Live metadata unavailable right now — check the official page.</span>
              )}
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}
