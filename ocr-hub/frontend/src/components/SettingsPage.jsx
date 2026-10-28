import React, { useEffect, useState } from 'react'
import { api } from '../api'

export default function SettingsPage() {
  const [s, setS] = useState(null)
  const [tokenInput, setTokenInput] = useState('')
  const [msg, setMsg] = useState('')

  useEffect(() => { api.settings().then(setS).catch((e) => alert(e.message)) }, [])
  if (!s) return <p className="text-sm" style={{ color: 'var(--ink-soft)' }}>Loading…</p>

  const save = async (patch) => {
    const cur = await api.saveSettings(patch)
    setS(cur)
    setMsg('Saved.')
    setTimeout(() => setMsg(''), 1500)
  }

  return (
    <>
      <div className="page-head"><h2>Settings</h2></div>
      <p className="page-sub mb-6">Stored locally in <code className="mono">data/config.json</code> on this machine.</p>

      <div className="max-w-xl flex flex-col gap-4">
        <section className="mcard">
          <h3 className="text-[15px] font-semibold">HuggingFace token</h3>
          <p className="notes">Removes download rate limits and unlocks gated models. Only this machine sees it.</p>
          <div className="flex gap-2 mt-1">
            <input
              className="input mono flex-1"
              placeholder={s.hf_token || 'paste token — hf_…'}
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
            />
            <button className="btn primary" onClick={() => save({ hf_token: tokenInput })}>Save</button>
          </div>
          <a className="text-xs underline" style={{ color: 'var(--azure)' }}
            href="https://huggingface.co/settings/tokens" target="_blank">Get a token</a>
        </section>

        <section className="mcard">
          <h3 className="text-[15px] font-semibold">Preferred device</h3>
          <select className="input mt-1" value={s.device_preference}
            onChange={(e) => save({ device_preference: e.target.value })}>
            <option value="auto">Auto — best available</option>
            <option value="cuda">CUDA (NVIDIA)</option>
            <option value="mps">MPS (Apple GPU)</option>
            <option value="cpu">CPU</option>
          </select>
        </section>

        <section className="mcard">
          <h3 className="text-[15px] font-semibold">OXR cloud parsing</h3>
          <p className="notes">
            Lets the SoMark OXR-1.0 card run by sending your files to <code className="mono">free-oxr.somark.ai</code>'s free API.
            Nothing is stored locally for it — and your documents leave this machine when it's on.
          </p>
          <label className="flex items-center gap-2 mt-2 text-sm cursor-pointer">
            <input type="checkbox" checked={!!s.oxr_cloud_enabled} onChange={(e) => save({ oxr_cloud_enabled: e.target.checked })} />
            Enable OXR cloud (free API)
          </label>
        </section>

        <StoragePanel />

        {msg && <p className="text-sm" style={{ color: 'var(--scan)' }}>{msg}</p>}
      </div>
    </>
  )
}

function StoragePanel() {
  const [st, setSt] = useState(null)
  const load = () => fetch('/api/storage').then((r) => r.json()).then(setSt)
  useEffect(() => { load() }, [])

  if (!st) return null
  return (
    <section className="mcard">
      <h3 className="text-[15px] font-semibold">Storage</h3>
      <p className="notes">{st.total_gb} GB of models on this machine. Nothing here is downloaded without your say-so.</p>
      {st.models.map((m) => (
        <div key={m.repo} className="flex items-center gap-2 mt-1 text-sm">
          <code className="mono text-xs" style={{ color: 'var(--ink-soft)' }}>{m.repo}</code>
          <span className="mono text-xs">{m.gb} GB</span>
          <button
            className="btn ghost !py-0.5 !px-2 text-xs ml-auto"
            onClick={async () => {
              await fetch(`/api/models/${encodeURIComponent(m.repo)}`, { method: 'DELETE' })
              load()
            }}
          >
            Delete
          </button>
        </div>
      ))}
      {st.models.length === 0 && (
        <p className="text-xs mt-1" style={{ color: 'var(--ink-soft)' }}>No models installed.</p>
      )}
    </section>
  )
}
