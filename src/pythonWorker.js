import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs'
import chatCoreSource from '../chat_core.py?raw'
import researchLabSource from '../research_lab.py?raw'

const PYODIDE_INDEX_URL = 'https://cdn.jsdelivr.net/pyodide/v314.0.7/full/'
const listenFiles = import.meta.glob('../docs_listen/*.txt', { eager: true, query: '?raw', import: 'default' })
const speechFiles = import.meta.glob('../docs_speech/*.txt', { eager: true, query: '?raw', import: 'default' })
const exclusionFiles = import.meta.glob('../docs_not/*.txt', { eager: true, query: '?raw', import: 'default' })

let pyodidePromise

function filesIn(files) {
  return Object.entries(files).map(([path, content]) => ({
    name: path.split('/').at(-1),
    content,
  }))
}

async function initializePython() {
  const pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX_URL })
  pyodide.FS.mkdirTree('/app/docs_listen')
  pyodide.FS.mkdirTree('/app/docs_speech')
  pyodide.FS.mkdirTree('/app/docs_not')
  pyodide.FS.writeFile('/app/research_lab.py', researchLabSource)
  pyodide.FS.writeFile('/app/chat_core.py', chatCoreSource)

  for (const [directory, files] of [
    ['docs_listen', listenFiles],
    ['docs_speech', speechFiles],
    ['docs_not', exclusionFiles],
  ]) {
    for (const file of filesIn(files)) {
      pyodide.FS.writeFile(`/app/${directory}/${file.name}`, file.content)
    }
  }

  pyodide.runPython("import sys; sys.path.insert(0, '/app'); import research_lab, chat_core")
  return pyodide
}

async function runPython(action, question = '') {
  pyodidePromise ??= initializePython()
  const pyodide = await pyodidePromise
  const globals = pyodide.toPy({ question })
  try {
    const source = action === 'brief'
      ? `import json\nfrom pathlib import Path\nimport research_lab\njson.dumps(research_lab.build_brief([Path('/app/docs_listen'), Path('/app/docs_speech')], Path('/app/docs_not')), ensure_ascii=False)`
      : `import json\nfrom pathlib import Path\nimport chat_core\nspeech = chat_core.load_speech_documents(Path('/app/docs_speech'))\nlisten = chat_core.load_speech_documents(Path('/app/docs_listen'))\njson.dumps(chat_core.answer_question(question, speech, listen), ensure_ascii=False)`
    const jsonResult = await pyodide.runPythonAsync(source, { globals })
    return JSON.parse(jsonResult)
  } finally {
    globals.destroy()
  }
}

self.onmessage = async event => {
  const { id, action, question } = event.data
  try {
    const result = await runPython(action, question)
    self.postMessage({ id, result })
  } catch (error) {
    self.postMessage({ id, error: error?.message || String(error) })
  }
}
