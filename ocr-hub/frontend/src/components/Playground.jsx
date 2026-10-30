import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api'

const COLORS = ['#0f9d8f', '#2f5fef', '#d97a06', '#7c3aed', '#b42318', '#0f766e', '#a21caf']

function fileToB64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader()
    r.onload = () => res(r.result)
    r.onerror = rej
    r.readAsDataURL(file)
  })
}

function AnnotatedImage({ url, blocks, showBoxes }) {
  const [dims, setDims] = useState({ w: 0, h: 0 })
  if (!url) return null
  return (
    <div className="relative inline-block max-w-full">
      <img src={url} className="max-w-full rounded" style={{ maxHeight: 480 }}
        onLoad={(e) => setDims({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
        onError={(e) => (e.target.style.opacity = 0)} />
      {showBoxes && dims.w > 0 &&
        (blocks || []).filter((b) => b.box).map((b, i) => {
          const [x0, y0, x1, y1] = b.box
          const c = COLORS[i % COLORS.length]
          return (
            <div key={i} title={`${b.text}${b.confidence != null ? ` · ${Math.round(b.confidence * 100)}%` : ''}`}
              style={{
                position: 'absolute',
                left: `${(x0 / dims.w) * 100}%`, top: `${(y0 / dims.h) * 100}%`,
                width: `${((x1 - x0) / dims.w) * 100}%`, height: `${((y1 - y0) / dims.h) * 100}%`,
                border: `1.5px solid ${c}`,
                background: `color-mix(in srgb, ${c} 12%, transparent)`,
                pointerEvents: 'auto',
              }} />
          )
        })}
    </div>
  )
}

function isRunning(r) { return r._status === 'pending' || r._status === 'running' }

function LoadingCard({ result }) {
  const live = result._status === 'running'
  return (
    <div className="result loading-glass" aria-busy="true">
      <div className="result-head">
        <b>{result.name}</b>
        <span className="elapsed">{live ? 'running…' : 'queued…'}</span>
      </div>
      <div className="shimmer" />
      <p className="text-sm mono" style={{ color: 'var(--ink-soft)' }}>
        {live
          ? (result.message || 'Model is reading the page — results land here the moment they finish.')
          : 'Waiting for a runner slot…'}
      </p>
    </div>
  )
}

function ResultCard({ result, pageImg, onRetry, retrying }) {
  const [view, setView] = useState('overlay')
  const color = COLORS[(result.model_id?.length || 0) % COLORS.length]
  const n = (result.blocks || []).filter((b) => b.text).length
  const fullText = (result.blocks || []).map((b) => b.text).join('\n')
  if (isRunning(result)) return <LoadingCard result={result} />

  return (
    <div className={`result ${result.error ? 'has-error' : ''}`}>
      <div className="result-head">
        <span style={{
          width: 9, height: 9, borderRadius: 999,
          background: result.error ? '#b42318' : color, display: 'inline-block', flexShrink: 0,
        }} />
        <b title={result.model_id}>{result.name}</b>
        <span className="mono text-[11px]" style={{ color: 'var(--ink-soft)' }}>{result.device || ''}</span>
        <span className="elapsed">{result.elapsed_s != null ? `${result.elapsed_s}s · ${n} block${n === 1 ? '' : 's'}` : ''}</span>
        {(result.error || result.elapsed_s != null) && onRetry && (
          <button className="btn ghost !py-0.5 !px-2 text-xs" disabled={retrying} onClick={onRetry}>
            {retrying ? 'Retrying…' : '♻ Retry'}
          </button>
        )}
      </div>
      {result.error && (
        <div className="err">
          {result.error}
          <button className="btn ghost !py-0.5 !px-2 text-xs mt-1 block"
            onClick={() => navigator.clipboard.writeText(`${result.name}:\n${result.error}`)}>
            Copy diagnostics
          </button>
        </div>
      )}
      <div className="viewbar">
        {['overlay', 'text', 'json'].map((v) => (
          <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{v}</button>
        ))}
        {!result.error && fullText && (
          <>
            <button onClick={() => navigator.clipboard.writeText(fullText)}>copy text</button>
            <button onClick={() => {
              const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
              const a = document.createElement('a')
              a.href = URL.createObjectURL(blob)
              a.download = `${result.model_id.replaceAll('/', '_')}.json`
              a.click()
            }}>export</button>
          </>
        )}
      </div>
      {view === 'overlay' && (
        <div className="overflow-auto text-center">
          <AnnotatedImage url={pageImg} blocks={result.blocks} showBoxes />
        </div>
      )}
      {view === 'text' && <pre className="plain whitespace-pre-wrap">{fullText || '—'}</pre>}
      {view === 'json' && <pre className="plain">{JSON.stringify(result, null, 2)}</pre>}
    </div>
  )
}

export default function Playground({ models, devices, tabVisible }) {
  const [selected, setSelected] = useState([])
  const [device, setDevice] = useState('auto')
  const [source, setSource] = useState(null)
  const [dragover, setDragover] = useState(false)
  const [pages, setPages] = useState(null) // array of {page, image, results, nPages}
  const [isPdf, setIsPdf] = useState(false)
  const [deviceResolved, setDeviceResolved] = useState(null)
  const [activePage, setActivePage] = useState(0)
  const [runId, setRunId] = useState(null)
  const [running, setRunning] = useState(false)
  const [retryMap, setRetryMap] = useState({}) // "page:model_id" -> bool
  const fileRef = useRef(null)

  const usable = models.filter((m) => m.installed)
  const devOpts = devices ? Object.entries(devices.devices).map(([k, d]) => ({ k, ...d })) : []

  /* prefill selection when arriving via "#playground:<hfId>" */
  useEffect(() => {
    const pre = window.__prefillModel
    if (pre) { window.__prefillModel = null; setSelected((s) => s.includes(pre) ? s : [...s, pre]) }
  }, [])

  /* live results over WS — card flips the moment its job finishes */
  useEffect(() => {
    if (!runId) return
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
    ws.onmessage = (e) => {
      try {
        const evt = JSON.parse(e.data)
        if (evt.type !== 'run' || evt.run_id !== runId) return
        if (evt.status === 'complete') { setRunning(false); return }
        setPages((ps) => {
          if (!ps) return ps
          const idx = (evt.page || 1) - 1
          return ps.map((p, i) => {
            if (i !== idx) return p
            return {
              ...p,
              results: p.results.map((r) => {
                if (r.model_id !== evt.model_id) return r
                if (evt.status === 'done' && evt.result?.blocks != null)
                  return { ...evt.result, _status: 'done' }
                if (evt.status === 'error' && evt.result?.blocks != null)
                  return { ...evt.result, _status: 'done' }
                if (evt.status === 'running')
                  return { ...r, _status: 'running', message: evt.message || '(working…)' }
                return { ...r }
              }),
            }
          })
        })
        // safety: if the batch-complete event was missed, finish the button
        // state as soon as every card on every page is done
        try {
          const allDone = ps.every((pg) => pg.results.every((rr) => !isRunning(rr)))
          if (allDone) setRunning(false)
        } catch {}
      } catch {}
    }
    return () => ws.close()
  }, [runId])

  const setFile = async (f) => {
    if (!f) return
    const pdf = f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
    setSource({ b64: await fileToB64(f), name: f.name, isPdf: pdf })
    setPages(null)
  }

  const submit = (payload) =>
    api.run({ image: payload.b64 ?? source.b64, model_ids: payload.model_ids, device: payload.device ?? device })

  /* placeholder pages so each selected model gets its own live card */
  const pendingPages = (ids, n) =>
    Array.from({ length: n }, (_, i) => ({
      page: i + 1, image: null,
      results: ids.map((id) => {
        const card = models.find((m) => m.hf_id === id)
        return { model_id: id, name: card?.name || id, _status: 'pending', blocks: [] }
      }),
    }))

  const startRun = async () => {
    if (!source?.b64 || !selected.length) return
    setRetryMap({}); setRunning(true)
    // optimistic live cards
    setPages(pendingPages(selected, 1)); setIsPdf(source.isPdf); setActivePage(0)
    try {
      const res = await submit({ model_ids: selected })
      setDeviceResolved(res.device); setRunId(res.run_id)
      setIsPdf(res.is_pdf)
      // real page images immediately, but cards stay pending — each flips to
      // done individually as its WS event arrives
      setPages(res.pages.map((p) => ({
        page: p.page,
        image: p.image,
        results: selected.map((id) => {
          const card = models.find((m) => m.hf_id === id)
          return { model_id: id, name: card?.name || id, _status: 'pending', blocks: [] }
        }),
      })))
      setActivePage(0)
    } catch (e) {
      setPages(null)
      setRunning(false)
      window.appToast?.(`!Run failed: ${e.message}`, 'err')
    }
  }

  const retryModel = async (pageNo, modelId) => {
    const key = `${pageNo}:${modelId}`
    setRetryMap((m) => ({ ...m, [key]: true }))
    try {
      const res = await fetch('/api/run/single', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_file: pageImageName(pageNo), model_ids: [modelId], device, page: pageNo, run_id: runId }),
      }).then((r) => r.json())
      const fresh = res.pages[0]
      setPages((ps) => ps.map((p) => (p.page === pageNo ? { ...p, results: p.results.map((r) => (r.model_id === modelId ? { ...fresh.results[0], _status: 'done' } : r)) } : p)))
    } catch (e) {
      window.appToast?.(`!Retry failed: ${e.message}`, 'err')
    }
    setRetryMap((m) => ({ ...m, [key]: false }))
  }

  const rerunPage = async (pageNo) => {
    setRetryMap((m) => ({ ...m, [`page${pageNo}`]: true }))
    setPages((ps) => ps.map((p) =>
      p.page === pageNo
        ? { ...p, results: p.results.map((r) => ({ ...r, _status: 'pending', error: null, blocks: [] })) }
        : p))
    try {
      const res = await fetch('/api/run/single', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_file: pageImageName(pageNo), model_ids: selected, device, page: pageNo, run_id: runId }),
      }).then((r) => r.json())
      const fresh = res.pages[0]
      setPages((ps) => ps.map((p) => (p.page === pageNo ? { ...p, results: fresh.results.map((r) => ({ ...r, _status: 'done' })) } : p)))
    } catch (e) {
      window.appToast?.(`!Retry failed: ${e.message}`, 'err')
    }
    setRetryMap((m) => ({ ...m, [`page${pageNo}`]: false }))
  }

  const pageImageName = (pageNo) => {
    const p = (pages || []).find((x) => x.page === pageNo)
    if (!p) return ''
    if (p.image) return p.image.split('/').pop()
    return (pages.find((x) => x.image)?.image || '').split('/').pop() ||
      (pages[0]?.image || '').split('/').pop()
  }

  const page = pages?.[activePage]
  const pageImg = page?.image ? `/api/results/${page.image.split('/').pop()}` : (pages && pages.find((x) => x.image) ? `/api/results/${pages.find((x) => x.image).image.split('/').pop()}` : null)

  return (
    <>
      <div className="page-head"><h2>Playground</h2></div>
      <p className="page-sub mb-4">Drop an image or PDF, pick models, compare side by side.</p>

      <div
        className={`mcard mb-6 flex flex-col gap-4`}
        style={dragover ? { borderColor: 'var(--azure)', boxShadow: '0 0 0 3px rgba(47,95,239,.15)' } : {}}
        onDragOver={(e) => { e.preventDefault(); setDragover(true) }}
        onDragLeave={() => setDragover(false)}
        onDrop={(e) => { e.preventDefault(); setDragover(false); setFile(e.dataTransfer.files?.[0]) }}
      >
        <div className="flex flex-wrap items-center gap-3">
          {source ? (
            <span className="text-sm font-medium">{source.isPdf ? '📄' : '🖼'} {source.name}</span>
          ) : (
            <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>Drag an image / PDF here, or</span>
          )}
          <button className="btn ghost" onClick={() => fileRef.current.click()}>Choose…</button>
          <input ref={fileRef} type="file" accept="image/*,application/pdf" hidden
            onChange={(e) => setFile(e.target.files?.[0])} />
          {source && (
            <button className="btn ghost" onClick={() => { setSource(null); setPages(null) }}>Remove file</button>
          )}
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <b className="text-sm">Models</b>
            {usable.length === 0 && (
              <span className="text-sm" style={{ color: 'var(--ink-soft)' }}>Nothing installed yet — grab one from the Catalog.</span>
            )}
          </div>
          {(() => {
            const order = [
              'Commercial OCR APIs (API Required)',
              'Vision-Language OCR (VLM-based)',
              'Document Parsing and Intelligence Frameworks',
              'Scientific Document OCR',
              'Transformer-Based OCR',
              'Fast / Lightweight OCR',
              'Traditional OCR Engines',
              'PDF Extraction and OCR Utilities',
              'Other OCR Models and Tools',
            ]
            const groups = new Map(usable.map((m) => [m.category || 'Other', []]))
            for (const m of usable) groups.get(m.category || 'Other').push(m)
            const sorted = [...groups.entries()].sort((a, b) =>
              (order.indexOf(a[0]) + 1 || 99) - (order.indexOf(b[0]) + 1 || 99))
            return sorted.map(([cat, ms]) => (
              <div key={cat} className="flex flex-wrap items-center gap-2">
                <span className="mono text-[11px]" style={{ color: 'var(--ink-soft)', minWidth: 110 }}>
                  {cat.replace(' (API Required)', '')}
                </span>
                {ms.map((m) => (
                  <button key={m.hf_id} className={`chip ${selected.includes(m.hf_id) ? 'on' : ''}`}
                    onClick={() => setSelected((s) => s.includes(m.hf_id) ? s.filter((x) => x !== m.hf_id) : [...s, m.hf_id])}>
                    {m.name}
                  </button>
                ))}
              </div>
            ))
          })()}
          <select className="input !w-48" value={device} onChange={(e) => setDevice(e.target.value)}>
            <option value="auto">Auto</option>
            {devOpts.map((d) => (
              <option key={d.k} value={d.k} disabled={!d.available}>
                {d.label}{d.available ? '' : ' — unavailable'}
              </option>
            ))}
          </select>
          <button className="btn primary ml-auto" onClick={startRun} disabled={running || !selected.length || !source}>
            {running ? 'Running…' : `Run ${selected.length || ''} models`}
          </button>
        </div>
      </div>

      {pages && pages.length > 1 && (
        <div className="mb-4 flex flex-wrap items-center gap-2">
          {pages.map((p, i) => (
            <button key={i} className={`chip ${i === activePage ? 'on' : ''}`} onClick={() => setActivePage(i)}>
              Page {p.page}
              {p.results.some((r) => r.error) && <span style={{ color: '#b42318' }}> ⚠</span>}
            </button>
          ))}
          <button className="btn ghost !py-0.5 ml-2" disabled={retryMap[`page${pages[activePage].page}`]}
            onClick={() => rerunPage(pages[activePage].page)}>
            {retryMap[`page${pages[activePage].page}`] ? 'Re-running page…' : `♻ Re-run page ${pages[activePage].page}`}
          </button>
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        {page?.results?.map((r) => (
          <ResultCard
            key={r.model_id}
            result={{ ...r, device: r.device || deviceResolved }}
            pageImg={pageImg}
            onRetry={() => retryModel(page.page, r.model_id)}
            retrying={retryMap[`${page.page}:${r.model_id}`]}
          />
        ))}
      </div>

      {!source && !pages && (
        <div className="mcard mb-6 text-center text-sm" style={{ color: 'var(--ink-soft)' }}>
          Every selected model reads the same page in parallel — each card lights up as soon as
          its result is ready. Retries run just the failed model, not the whole batch.
        </div>
      )}
    </>
  )
}
