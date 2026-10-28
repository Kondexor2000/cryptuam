import { useState } from 'react'
import { ArrowDown, ArrowRight, ArrowUpRight, AudioLines, BookOpen, BrainCircuit, CircleHelp, FileText, FlaskConical, LoaderCircle, Menu, Network, Send, Sparkles, X } from 'lucide-react'
import './api.css'

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'https://cryptuam-python-api.onrender.com').replace(/\/$/, '')

const sections = [
  { id: 'projekt', label: 'Projekt' },
  { id: 'narzedzia', label: 'Narzędzia' },
  { id: 'aplikacja', label: 'Demo' },
  { id: 'eksperyment', label: 'Eksperyment' },
]

function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('Mieszana')
  const [questionOpen, setQuestionOpen] = useState(false)
  const [chatQuestion, setChatQuestion] = useState('')
  const [chatResult, setChatResult] = useState(null)
  const [brief, setBrief] = useState(null)
  const [chatLoading, setChatLoading] = useState(false)
  const [briefLoading, setBriefLoading] = useState(false)
  const [apiError, setApiError] = useState('')

  async function askChat(event) {
    event.preventDefault()
    if (!chatQuestion.trim()) return
    setChatLoading(true)
    setApiError('')
    setChatResult(null)
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: chatQuestion.trim() }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Nie udało się uzyskać odpowiedzi.')
      setChatResult(payload)
    } catch (error) {
      setApiError(`Nie mogę połączyć się z API Pythona. ${error.message}`)
    } finally {
      setChatLoading(false)
    }
  }

  async function loadBrief() {
    setBriefLoading(true)
    setApiError('')
    try {
      const response = await fetch(`${API_BASE_URL}/api/brief`)
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail || 'Nie udało się zbudować briefu.')
      setBrief(payload)
    } catch (error) {
      setApiError(`Nie mogę połączyć się z API Pythona. ${error.message}`)
    } finally {
      setBriefLoading(false)
    }
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#projekt" aria-label="AI Research Lab — strona główna">
          <span className="brand-mark"><FlaskConical size={17} strokeWidth={1.8} /></span>
          <span>AI RESEARCH <i>LAB</i></span>
        </a>
        <button className="mobile-menu" onClick={() => setMenuOpen(!menuOpen)} aria-label={menuOpen ? 'Zamknij menu' : 'Otwórz menu'}>{menuOpen ? <X size={20} /> : <Menu size={20} />}</button>
        <nav className={menuOpen ? 'nav open' : 'nav'} aria-label="Nawigacja główna">
          {sections.map(({ id, label }) => <a key={id} href={`#${id}`} onClick={() => setMenuOpen(false)}>{label}</a>)}
          <a className="github-link" href="https://github.com/Kondexor2000/cryptuam" target="_blank" rel="noreferrer">Zobacz repozytorium <ArrowUpRight size={15} /></a>
        </nav>
      </header>

      <section className="hero" id="projekt">
        <div className="hero-copy">
          <div className="eyebrow"><span className="pulse" /> PROTOTYP BADAWCZY <span className="eyebrow-divider">/</span> UCZENIE JĘZYKÓW</div>
          <h1>Lepsze pytania.<br /><span>Uważniejsze</span><br />eksperymenty.</h1>
          <p className="hero-lede">Małe laboratorium na styku języka i AI. Porządkuje materiał do nauki, proponuje eksperymenty i pomaga przyjrzeć się temu, jak uczymy się słów.</p>
          <div className="hero-actions"><a className="button button-dark" href="#narzedzia">Poznaj projekt <ArrowDown size={16} /></a><a className="text-link" href="#eksperyment">Zobacz pytanie badawcze <ArrowRight size={15} /></a></div>
          <div className="hero-footnote"><span>01 — 04</span><span>MAŁE NARZĘDZIA, WIĘKSZA CIEKAWOŚĆ</span></div>
        </div>
        <div className="hero-art" aria-label="Abstrakcyjna wizualizacja danych i języka">
          <div className="art-label art-label-top"><span>FIG 01</span><span>LEARNING SYSTEMS</span></div>
          <div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" />
          <div className="core"><Network size={40} strokeWidth={1.1} /><span>język</span></div>
          <span className="node node-a">słuchanie</span><span className="node node-b">mówienie</span><span className="node node-c">pamięć</span><span className="node node-d">powtórki</span>
          <span className="art-cross cross-a">+</span><span className="art-cross cross-b">+</span><span className="art-dot dot-a" /><span className="art-dot dot-b" />
          <div className="art-label art-label-bottom"><span>WEJŚCIE</span><span className="art-line" /><span>HIPOTEZA</span><span className="art-line" /><span>OBSERWACJA</span></div>
        </div>
      </section>

      <section className="ticker" aria-label="Obszary projektu"><div className="ticker-track"><span>PRZETWARZANIE JĘZYKA</span><span className="ticker-star">✳</span><span>EKSPERYMENTY UCZENIA</span><span className="ticker-star">✳</span><span>ODPOWIEDZIALNE AI</span><span className="ticker-star">✳</span><span>PRZETWARZANIE JĘZYKA</span><span className="ticker-star">✳</span><span>EKSPERYMENTY UCZENIA</span><span className="ticker-star">✳</span><span>ODPOWIEDZIALNE AI</span><span className="ticker-star">✳</span></div></section>

      <section className="intro section-wrap" id="narzedzia">
        <div className="section-kicker"><span>01</span> / CO TU ZNAJDZIESZ</div>
        <div className="intro-grid"><h2>Warsztat, który<br />zaczyna się od <em>ciekawości.</em></h2><div className="intro-copy"><p>To eksperymentalny zestaw narzędzi do pracy z materiałem językowym. Zamiast obiecywać gotowe odpowiedzi, pomaga układać dane i zamieniać intuicje w pytania, które można sprawdzić.</p><p>React wyświetla stronę, a API w Pythonie obsługuje chatbot i generator briefu. Narzędzia czytają pliki projektu i nie wymagają bazy danych.</p></div></div>
        <div className="cards">
          <article className="tool-card card-sand"><div className="card-top"><span className="card-icon"><AudioLines size={19} /></span><span className="card-index">NARZĘDZIE 01</span></div><h3>Chatbot<br />do rozmowy</h3><p>Eksperymentalny interfejs, który korzysta z materiałów językowych i pomaga eksplorować wpisane zagadnienia.</p><div className="card-bottom"><span>TEKST → KONTEKST</span><ArrowUpRight size={17} /></div></article>
          <article className="tool-card card-lilac"><div className="card-top"><span className="card-icon"><FlaskConical size={19} /></span><span className="card-index">NARZĘDZIE 02</span></div><h3>Brief<br />badawczy</h3><p>Porządkuje materiał źródłowy i wykluczenia, a następnie szkicuje hipotezę oraz plan eksperymentu.</p><div className="card-bottom"><span>DANE → PYTANIE</span><ArrowUpRight size={17} /></div></article>
          <article className="tool-card card-mint"><div className="card-top"><span className="card-icon"><BrainCircuit size={19} /></span><span className="card-index">PERSPEKTYWA</span></div><h3>Proces<br />nad obietnicą</h3><p>Propozycja eksperymentu to początek rozmowy. Projekt nie twierdzi, że uzyskał potwierdzone wyniki.</p><div className="card-bottom"><span>HIPOTEZA → WERYFIKACJA</span><ArrowUpRight size={17} /></div></article>
        </div>
      </section>

      <section className="live-tools section-wrap" id="aplikacja">
        <div className="section-kicker"><span>02</span> / ŻYWE DEMO Z PYTHONA</div>
        <div className="live-heading"><div><h2>Wypróbuj <em>narzędzia.</em></h2><p>Te formularze wysyłają pytanie do API w Pythonie. Chatbot szuka kontekstu w `docs_speech`, a brief powstaje z list słownictwa i wykluczeń w repozytorium.</p></div><span className="api-badge"><span /> REACT ↔ FASTAPI</span></div>
        <div className="live-grid">
          <article className="demo-card demo-chat"><div className="demo-card-head"><span className="demo-icon"><AudioLines size={18} /></span><span>PYTHON / CHATBOT</span></div><h3>Zadaj pytanie<br />materiałom.</h3><form className="chat-form" onSubmit={askChat}><label htmlFor="chat-question">Pytanie do dokumentów</label><div className="chat-input-row"><input id="chat-question" value={chatQuestion} onChange={event => setChatQuestion(event.target.value)} maxLength={500} placeholder="Np. co pomaga w nauce słownictwa?" /><button type="submit" disabled={chatLoading || !chatQuestion.trim()} aria-label="Wyślij pytanie">{chatLoading ? <LoaderCircle size={17} className="spin" /> : <Send size={16} />}</button></div></form>{chatResult && <div className="chat-result"><span>ODPOWIEDŹ · {chatResult.source}</span><p>{chatResult.answer}</p>{chatResult.ideas?.length > 0 && <div className="idea-tags">{chatResult.ideas.map(idea => <span key={idea}>{idea}</span>)}</div>}</div>}</article>
          <article className="demo-card demo-brief"><div className="demo-card-head"><span className="demo-icon"><FileText size={18} /></span><span>PYTHON / RESEARCH LAB</span></div><h3>Brief z bieżących<br />plików projektu.</h3><p className="demo-desc">Generator sam zlicza pozycje źródłowe, stosuje wykluczenia i zwraca hipotezę z planem badania.</p><button className="button brief-button" onClick={loadBrief} disabled={briefLoading}>{briefLoading ? <LoaderCircle size={16} className="spin" /> : <ArrowRight size={16} />}{briefLoading ? 'Tworzę brief…' : brief ? 'Odśwież brief' : 'Wygeneruj brief'}</button>{brief && <div className="brief-result"><span className="brief-status">{brief.status}</span><p className="brief-question">{brief.question}</p><div className="brief-stats"><div><strong>{brief.sources.entries_read}</strong><span>W CZYTANYCH LISTACH</span></div><div><strong>{brief.corpus.eligible_entries}</strong><span>KWALIFIKUJE SIĘ</span></div><div><strong>{brief.exclusions.entries_removed}</strong><span>WYKLUCZONO</span></div></div><p className="brief-hypothesis">{brief.hypothesis}</p></div>}</article>
        </div>
        {apiError && <p className="api-error" role="alert">{apiError} Przy pierwszym wywołaniu bezczynne darmowe API może potrzebować chwili, żeby się uruchomić.</p>}
        <p className="api-note">API działa bez bazy danych. Korzysta z plików zapisanych w repozytorium; darmowa instancja może zasypiać po okresie bez ruchu.</p>
      </section>

      <section className="experiment" id="eksperyment">
        <div className="section-wrap experiment-inner"><div className="section-kicker light"><span>03</span> / PYTANIE, KTÓRE NAPĘDZA PROJEKT</div><div className="experiment-grid"><div><div className="experiment-tag"><Sparkles size={13} /> HIPOTEZA EKSPERYMENTALNA</div><h2>Czy mieszana praktyka pomaga <em>pamiętać</em> dłużej?</h2><p className="experiment-intro">Propozycja sprawdza, czy przeplatanie słownictwa z różnych źródeł daje lepsze wyniki w odroczonym przypominaniu niż nauka plik po pliku.</p><button className="question-trigger" onClick={() => setQuestionOpen(!questionOpen)}><CircleHelp size={16} /> {questionOpen ? 'Ukryj założenia' : 'Jak rozumieć to pytanie?'} <span>{questionOpen ? '−' : '+'}</span></button>{questionOpen && <div className="question-answer">To pytanie badawcze wygenerowane na podstawie lokalnych list słownictwa. Sam projekt nie przeprowadził badania — wynik wymaga zaplanowania i wykonania eksperymentu z uczestnikami.</div>}</div>
          <div className="experiment-panel"><div className="panel-head"><span>SCHEMAT BADANIA</span><span className="panel-status"><span /> PROPOZYCJA</span></div><div className="tabs" role="tablist" aria-label="Warianty ćwiczenia">{['Mieszana', 'Plik po pliku'].map(tab => <button key={tab} role="tab" aria-selected={activeTab === tab} className={activeTab === tab ? 'tab active' : 'tab'} onClick={() => setActiveTab(tab)}>{tab}</button>)}</div><div className="study-flow"><div className="flow-label"><span>01</span> SPOSÓB NAUKI</div><div className="flow-items">{(activeTab === 'Mieszana' ? ['plik A', 'plik B', 'plik C', 'plik A', 'plik B', 'plik C'] : ['plik A', 'plik A', 'plik A', 'plik B', 'plik B', 'plik C']).map((item, index) => <span key={`${item}-${index}`} className={`word-chip chip-${item.slice(-1).toLowerCase()}`}>{item}</span>)}</div><div className="flow-divider"><span /><ArrowDown size={15} /><span /></div><div className="flow-label"><span>02</span> POMIAR</div><div className="measure-row"><div><BookOpen size={16} /><span>Przypominanie</span></div><span className="measure-time">od razu + po czasie</span></div></div><div className="panel-foot"><span>STAŁY CZAS I EKSPOZYCJA</span><span>01 / 02</span></div></div></div></div>
      </section>

      <section className="principles section-wrap"><div className="section-kicker"><span>04</span> / DOBRZE WIEDZIEĆ</div><div className="principles-grid"><h2>Przejrzyste<br /><em>ograniczenia.</em></h2><div className="principle-list"><div className="principle"><span className="principle-no">01</span><div><h3>To propozycja, nie wynik.</h3><p>Brief opisuje możliwy eksperyment; nie potwierdza skuteczności żadnej metody.</p></div><ArrowUpRight size={16} /></div><div className="principle"><span className="principle-no">02</span><div><h3>Bez bazy danych.</h3><p>API w Pythonie czyta pliki projektu podczas obsługi żądań. Nie zapisuje pytań ani wyników w bazie.</p></div><ArrowUpRight size={16} /></div><div className="principle"><span className="principle-no">03</span><div><h3>AI ma swoje granice.</h3><p>Chatbot korzysta z wyszukiwania kontekstowego w plikach, a nie z hostowanego dużego modelu językowego.</p></div><ArrowUpRight size={16} /></div></div></div></section>

      <footer className="footer"><div className="footer-top"><a className="brand footer-brand" href="#projekt"><span className="brand-mark"><FlaskConical size={17} /></span><span>AI RESEARCH <i>LAB</i></span></a><span className="footer-note">EKSPERYMENTUJ Z CIEKAWOŚCIĄ. WERYFIKUJ Z UWAGĄ.</span><a className="footer-github" href="https://github.com/Kondexor2000/cryptuam" target="_blank" rel="noreferrer">GITHUB <ArrowUpRight size={15} /></a></div><div className="footer-bottom"><span>PROTOTYP BADAWCZY · 2026</span><a href="#projekt">DO GÓRY ↑</a><span>WYTWORZONE Z CIEKAWOŚCI</span></div></footer>
    </main>
  )
}

export default App
