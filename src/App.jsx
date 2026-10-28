import { useState } from 'react'
import { ArrowDown, ArrowRight, ArrowUpRight, AudioLines, BookOpen, BrainCircuit, CircleHelp, FlaskConical, Menu, Network, Sparkles, X } from 'lucide-react'

const sections = [
  { id: 'projekt', label: 'Projekt' },
  { id: 'narzedzia', label: 'Narzędzia' },
  { id: 'eksperyment', label: 'Eksperyment' },
]

function App() {
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeTab, setActiveTab] = useState('Mieszana')
  const [questionOpen, setQuestionOpen] = useState(false)

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
          <div className="hero-footnote"><span>01 — 03</span><span>MAŁE NARZĘDZIA, WIĘKSZA CIEKAWOŚĆ</span></div>
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
        <div className="intro-grid"><h2>Warsztat, który<br />zaczyna się od <em>ciekawości.</em></h2><div className="intro-copy"><p>To eksperymentalny zestaw narzędzi do pracy z materiałem językowym. Zamiast obiecywać gotowe odpowiedzi, pomaga układać dane i zamieniać intuicje w pytania, które można sprawdzić.</p><p>Projekt łączy prosty chatbot z generatorem briefu badawczego. Całość działa na lokalnych plikach i nie wymaga bazy danych.</p></div></div>
        <div className="cards">
          <article className="tool-card card-sand"><div className="card-top"><span className="card-icon"><AudioLines size={19} /></span><span className="card-index">NARZĘDZIE 01</span></div><h3>Chatbot<br />do rozmowy</h3><p>Eksperymentalny interfejs, który korzysta z materiałów językowych i pomaga eksplorować wpisane zagadnienia.</p><div className="card-bottom"><span>TEKST → KONTEKST</span><ArrowUpRight size={17} /></div></article>
          <article className="tool-card card-lilac"><div className="card-top"><span className="card-icon"><FlaskConical size={19} /></span><span className="card-index">NARZĘDZIE 02</span></div><h3>Brief<br />badawczy</h3><p>Porządkuje materiał źródłowy i wykluczenia, a następnie szkicuje hipotezę oraz plan eksperymentu.</p><div className="card-bottom"><span>DANE → PYTANIE</span><ArrowUpRight size={17} /></div></article>
          <article className="tool-card card-mint"><div className="card-top"><span className="card-icon"><BrainCircuit size={19} /></span><span className="card-index">PERSPEKTYWA</span></div><h3>Proces<br />nad obietnicą</h3><p>Propozycja eksperymentu to początek rozmowy. Projekt nie twierdzi, że uzyskał potwierdzone wyniki.</p><div className="card-bottom"><span>HIPOTEZA → WERYFIKACJA</span><ArrowUpRight size={17} /></div></article>
        </div>
      </section>

      <section className="experiment" id="eksperyment">
        <div className="section-wrap experiment-inner"><div className="section-kicker light"><span>02</span> / PYTANIE, KTÓRE NAPĘDZA PROJEKT</div><div className="experiment-grid"><div><div className="experiment-tag"><Sparkles size={13} /> HIPOTEZA EKSPERYMENTALNA</div><h2>Czy mieszana praktyka pomaga <em>pamiętać</em> dłużej?</h2><p className="experiment-intro">Propozycja sprawdza, czy przeplatanie słownictwa z różnych źródeł daje lepsze wyniki w odroczonym przypominaniu niż nauka plik po pliku.</p><button className="question-trigger" onClick={() => setQuestionOpen(!questionOpen)}><CircleHelp size={16} /> {questionOpen ? 'Ukryj założenia' : 'Jak rozumieć to pytanie?'} <span>{questionOpen ? '−' : '+'}</span></button>{questionOpen && <div className="question-answer">To pytanie badawcze wygenerowane na podstawie lokalnych list słownictwa. Sam projekt nie przeprowadził badania — wynik wymaga zaplanowania i wykonania eksperymentu z uczestnikami.</div>}</div>
          <div className="experiment-panel"><div className="panel-head"><span>SCHEMAT BADANIA</span><span className="panel-status"><span /> PROPOZYCJA</span></div><div className="tabs" role="tablist" aria-label="Warianty ćwiczenia">{['Mieszana', 'Plik po pliku'].map(tab => <button key={tab} role="tab" aria-selected={activeTab === tab} className={activeTab === tab ? 'tab active' : 'tab'} onClick={() => setActiveTab(tab)}>{tab}</button>)}</div><div className="study-flow"><div className="flow-label"><span>01</span> SPOSÓB NAUKI</div><div className="flow-items">{(activeTab === 'Mieszana' ? ['plik A', 'plik B', 'plik C', 'plik A', 'plik B', 'plik C'] : ['plik A', 'plik A', 'plik A', 'plik B', 'plik B', 'plik C']).map((item, index) => <span key={`${item}-${index}`} className={`word-chip chip-${item.slice(-1).toLowerCase()}`}>{item}</span>)}</div><div className="flow-divider"><span /><ArrowDown size={15} /><span /></div><div className="flow-label"><span>02</span> POMIAR</div><div className="measure-row"><div><BookOpen size={16} /><span>Przypominanie</span></div><span className="measure-time">od razu + po czasie</span></div></div><div className="panel-foot"><span>STAŁY CZAS I EKSPOZYCJA</span><span>01 / 02</span></div></div></div></div>
      </section>

      <section className="principles section-wrap"><div className="section-kicker"><span>03</span> / DOBRZE WIEDZIEĆ</div><div className="principles-grid"><h2>Przejrzyste<br /><em>ograniczenia.</em></h2><div className="principle-list"><div className="principle"><span className="principle-no">01</span><div><h3>To propozycja, nie wynik.</h3><p>Brief opisuje możliwy eksperyment; nie potwierdza skuteczności żadnej metody.</p></div><ArrowUpRight size={16} /></div><div className="principle"><span className="principle-no">02</span><div><h3>Dane zostają lokalne.</h3><p>Narzędzia Pythona czytają lokalne pliki. Strona prezentuje projekt i nie ładuje tych materiałów.</p></div><ArrowUpRight size={16} /></div><div className="principle"><span className="principle-no">03</span><div><h3>AI ma swoje granice.</h3><p>Chatbot i modele są prototypowe. Odpowiedzi wymagają krytycznej oceny.</p></div><ArrowUpRight size={16} /></div></div></div></section>

      <footer className="footer"><div className="footer-top"><a className="brand footer-brand" href="#projekt"><span className="brand-mark"><FlaskConical size={17} /></span><span>AI RESEARCH <i>LAB</i></span></a><span className="footer-note">EKSPERYMENTUJ Z CIEKAWOŚCIĄ. WERYFIKUJ Z UWAGĄ.</span><a className="footer-github" href="https://github.com/Kondexor2000/cryptuam" target="_blank" rel="noreferrer">GITHUB <ArrowUpRight size={15} /></a></div><div className="footer-bottom"><span>PROTOTYP BADAWCZY · 2026</span><a href="#projekt">DO GÓRY ↑</a><span>WYTWORZONE Z CIEKAWOŚCI</span></div></footer>
    </main>
  )
}

export default App
