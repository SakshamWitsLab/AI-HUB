const j = async (r) => {
  if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.detail || r.statusText)
  return r.json()
}

export const api = {
  models: () => fetch('/api/models').then(j),
  preview: (repo) => fetch(`/api/models/preview?repo=${encodeURIComponent(repo)}`).then(j),
  add: (repo) => fetch(`/api/models/add?repo=${encodeURIComponent(repo)}`, { method: 'POST' }).then(j),
  download: (repo) => fetch(`/api/models/${encodeURIComponent(repo)}/download`, { method: 'POST' }).then(j),
  devices: () => fetch('/api/devices').then(j),
  settings: () => fetch('/api/settings').then(j),
  saveSettings: (patch) =>
    fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }).then(j),
  run: (body) =>
    fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(j),
}

export function imageToBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader()
    r.onload = () => res(r.result) // data URI — backend strips prefix
    r.onerror = rej
    r.readAsDataURL(file)
  })
}
