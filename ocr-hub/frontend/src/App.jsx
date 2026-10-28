import React, { Component, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import Catalog from './components/Catalog'
import Playground from './components/Playground'
import SettingsPage from './components/SettingsPage'

const TABS = [
  { id: 'catalog', label: 'Catalog', icon: '▤' },
  { id: 'playground', label: 'Playground', icon: '⦿' },
  { id: 'settings', label: 'Settings', icon: '⛭' },
]

/* --------------------------------------------------------------- toasts — */
let pushToast = () => {}
function ToastHost() {
  const [toasts, setToasts] = useState([])
  useEffect(() => {
    pushToast = (msg, kind = 'info') => {
      const id = Math.random().toString(36).slice(2)
      setToasts((t) => [...t.slice(-3), { id, msg, kind }])
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4500)
    }
    window.appToast = pushToast
    return () => { pushToast = () => {}; window.appToast = null }
  }, [])
  return (
    <div className="toast-stack">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>{t.msg}</div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------ error boundary — */
class ErrorBoundary extends Component {
  constructor(p) { super(p); this.state = { error: null } }
  static getDerivedStateFromError(error) { return { error } }
  componentDidCatch(error, info) { console.error('UI crash:', error, info) }
  render() {
    if (!this.state.error) return this.props.children
    const err = this.state.error
    return (
      <div className="mcard max-w-xl mx-auto" style={{ marginTop: '15vh' }}>
        <h3 className="text-lg font-semibold mb-1">The interface hit an error</h3>
        <pre className="err" style={{ whiteSpace: 'pre-wrap' }}>
          {err.message || String(err)}
        </pre>
        <button className="btn primary mt-3" onClick={() => this.setState({ error: null })}>
          Back to the app
        </button>
        <button className="btn ghost mt-3 ml-2" onClick={() => location.reload()}>Reload page</button>
      </div>
    )
  }
}

export default function App() {
  const [tab, setTab] = useState('catalog')
  const [devices, setDevices] = useState(null)
  const [models, setModels] = useState([])
  const [dl, setDl] = useState(null)
  const wsRef = useRef(null)
  const reload = () => api.models().then((d) => setModels(d.models)).catch(() => {})

  useEffect(() => {
    api.devices().then(setDevices).catch((e) => pushToast(`!Device check failed: ${e.message}`, 'err'))
    reload()
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
    ws.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data)
        if (evt.type === 'run') return
        if (evt.type !== 'download') return
        const short = evt.repo.split('/').pop()
        if (evt.status === 'running' || evt.status === 'queued') {
          setDl({ name: short, pct: evt.progress ?? 0 })
        } else if (evt.status === 'done') {
          pushToast(`${short} installed`)
          setDl(null); reload()
        } else if (evt.status === 'error') {
          pushToast(`!${short} failed: ${evt.error || 'unknown'}`, 'err')
          setDl(null); reload()
        }
      } catch {}
    }
    wsRef.current = ws
    window.addEventListener('unhandledrejection', (e) =>
      pushToast(`Unexpected error: ${(e.reason?.message ?? String(e.reason)).slice(0, 200)}`, 'err'))
    return () => ws.close()
  }, [])

  const installedCount = useMemo(
    () => models.filter((m) => m.installed && !m.hf_id.startsWith('builtin:')).length,
    [models],
  )
  const devList = useMemo(() => (devices ? Object.values(devices.devices) : []), [devices])

  return (
    <ErrorBoundary>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand"><h1>OCR</h1><span>HUB</span></div>
          {TABS.map((t) => (
            <button key={t.id} className={`nav-btn ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              <span aria-hidden>{t.icon}</span> {t.label}
              {t.id === 'catalog' && installedCount > 0 && <span className="count">{installedCount}</span>}
            </button>
          ))}
          <div className="sidebar-foot">
            <div>Devices</div>
            {devList.map((d) => (
              <div key={d.label} title={d.detail}>
                <span className={d.available ? 'pill-ok' : 'pill-no'}>{d.available ? '●' : 'inactive'}</span>{' '}
                {d.label.split(' ')[0]}
              </div>
            ))}
          </div>
        </aside>

        <main className="content">
          <div hidden={tab !== 'catalog'}><Catalog models={models} reload={reload} devices={devices} /></div>
          <div hidden={tab !== 'playground'}><Playground models={models} devices={devices} tabVisible={tab === 'playground'} /></div>
          <div hidden={tab !== 'settings'}><SettingsPage /></div>
        </main>
      </div>

      {dl && <div className="toast">⬇ {dl.name} {dl.pct}%</div>}
      <ToastHost />
    </ErrorBoundary>
  )
}
