let worker
let nextId = 0
const pending = new Map()

function getWorker() {
  if (worker) return worker

  worker = new Worker(new URL('./pythonWorker.js', import.meta.url), { type: 'module' })
  worker.onmessage = ({ data }) => {
    const resolvePending = pending.get(data.id)
    if (!resolvePending) return
    pending.delete(data.id)
    if (data.error) resolvePending.reject(new Error(data.error))
    else resolvePending.resolve(data.result)
  }
  worker.onerror = event => {
    for (const request of pending.values()) request.reject(new Error(event.message || 'Błąd uruchamiania Pythona.'))
    pending.clear()
    worker?.terminate()
    worker = undefined
  }
  return worker
}

export function runPython(action, question = '') {
  const id = ++nextId
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    getWorker().postMessage({ id, action, question })
  })
}
