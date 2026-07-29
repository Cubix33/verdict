import React, { useState, useEffect } from 'react';

function App() {
  const [imagePath, setImagePath] = useState('data/example.png');
  const [logs, setLogs] = useState([]);
  const [triageData, setTriageData] = useState(null);
  const [probeData, setProbeData] = useState(null);
  const [adjudicationData, setAdjudicationData] = useState(null);
  const [verificationData, setVerificationData] = useState(null);
  const [cards, setCards] = useState([]);
  const [activeStep, setActiveStep] = useState('idle'); // idle | triage | prober | adjudicator | verifier | done
  const [selectedCard, setSelectedCard] = useState(null);
  const [uploading, setUploading] = useState(false);

  // Auto-fetch manifest on start
  useEffect(() => {
    fetchManifest();
  }, []);

  function addLog(message, type = 'info') {
    const time = new Date().toLocaleTimeString();
    setLogs(prev => [{ time, message, type }, ...prev]);
  }

  async function fetchManifest() {
    try {
      const r = await fetch('http://localhost:8000/api/manifest');
      const j = await r.json();
      if (j.cards) {
        setCards(j.cards);
        if (j.cards.length > 0 && !selectedCard) {
          setSelectedCard(j.cards[0]);
        }
      }
    } catch (e) {
      addLog('Failed to fetch manifest: ' + String(e), 'error');
    }
  }

  async function runTriage() {
    setActiveStep('triage');
    addLog(`Initiating Triage Agent on: ${imagePath}`, 'triage');
    try {
      const res = await fetch('http://localhost:8000/api/triage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: imagePath })
      });
      const data = await res.json();
      setTriageData(data);
      if (data.decision === 'escalate') {
        addLog(`Triage Decision: ESCALATE - Reason: ${data.reason}`, 'warning');
      } else {
        addLog(`Triage Decision: RESOLVED - Reason: ${data.reason}`, 'success');
      }
    } catch (e) {
      addLog('Triage failed: ' + String(e), 'error');
    }
  }

  async function runProbes() {
    setActiveStep('prober');
    addLog(`Initiating Prober Agent on: ${imagePath}`, 'prober');
    try {
      const res = await fetch('http://localhost:8000/api/probe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: imagePath })
      });
      const data = await res.json();
      setProbeData(data);
      addLog(`Prober generated ${data.created_cards?.length || 0} evidence cards`, 'success');
      await fetchManifest();
    } catch (e) {
      addLog('Prober failed: ' + String(e), 'error');
    }
  }

  async function runAdjudication() {
    setActiveStep('adjudicator');
    addLog('Initiating Starved Adjudicator Agent. Image visibility: LOCKED', 'adjudicator');
    try {
      const res = await fetch('http://localhost:8000/api/adjudicate', { method: 'POST' });
      const data = await res.json();
      setAdjudicationData(data);
      const claimsCount = data.claims?.length || 0;
      addLog(`Adjudicator compiled claims list: ${claimsCount} claims found. Verdict: ${data.verdict}`, claimsCount > 0 ? 'warning' : 'success');
    } catch (e) {
      addLog('Adjudication failed: ' + String(e), 'error');
    }
  }

  async function runVerification() {
    setActiveStep('verifier');
    addLog('Initiating Context-Free Verifier Agent (Skeptic Mode)', 'verifier');
    try {
      const res = await fetch('http://localhost:8000/api/verify', { method: 'POST' });
      const data = await res.json();
      setVerificationData(data.verification);
      setAdjudicationData(data.adjudication); // Sync state
      
      const holds = data.verification.claims.filter(c => c.overall === 'holds').length;
      const breaks = data.verification.claims.filter(c => c.overall === 'broken').length;
      addLog(`Verification complete: ${holds} bonds HOLD, ${breaks} bonds BROKEN.`, breaks > 0 ? 'error' : 'success');
      setActiveStep('done');
    } catch (e) {
      addLog('Verification failed: ' + String(e), 'error');
      setActiveStep('done');
    }
  }

  async function clearWorkflow() {
    try {
      const res = await fetch('http://localhost:8000/api/reset', { method: 'POST' });
      if (res.ok) {
        addLog('Backend ledger database, cache, and crop files cleared.', 'success');
      } else {
        addLog('Failed to clear backend ledger data.', 'error');
      }
    } catch (e) {
      addLog('Error calling reset API: ' + String(e), 'error');
    }
    
    setTriageData(null);
    setProbeData(null);
    setAdjudicationData(null);
    setVerificationData(null);
    setCards([]);
    setSelectedCard(null);
    setActiveStep('idle');
    addLog('Workspace reset complete. Ready to audit a new receipt.', 'success');
  }

  async function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    addLog(`Uploading file: ${file.name}...`, 'info');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setImagePath(data.image_path);
        addLog(`File uploaded successfully: ${data.image_path}`, 'success');
      } else {
        addLog(`Upload failed: ${data.detail || 'unknown error'}`, 'error');
      }
    } catch (e) {
      addLog(`Upload error: ${String(e)}`, 'error');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div style={styles.container}>
      {/* Global CSS Inject */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        body { margin: 0; background-color: #0d0f12; color: #e2e8f0; font-family: 'Outfit', sans-serif; overflow-x: hidden; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0f1319; }
        ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4a5568; }
        .glow-btn:hover { box-shadow: 0 0 12px rgba(66, 153, 225, 0.4); transform: translateY(-1px); }
        .claim-card { border: 1px solid #1a222f; transition: all 0.2s ease; }
        .claim-card:hover { border-color: #3b82f6; box-shadow: 0 0 8px rgba(59, 130, 246, 0.2); }
      `}</style>

      {/* Header Banner */}
      <header style={styles.header}>
        <div style={styles.logoGroup}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
          <span style={styles.logoText}>VERDICT</span>
          <span style={styles.logoBadge}>AI Arena 3.0</span>
        </div>
        <div style={styles.systemStatus}>
          <div style={{ ...styles.statusDot, backgroundColor: '#10b981' }}></div>
          <span style={{ fontSize: '13px', color: '#94a3b8' }}>Network: Operational</span>
        </div>
      </header>

      {/* Dashboard Grid */}
      <main style={styles.mainGrid}>
        
        {/* Left Side: Submission & Controls */}
        <section style={styles.controlPanel}>
          <div style={styles.cardHeader}>
            <h2 style={styles.cardTitle}>Evidence Submission</h2>
            <p style={styles.cardSubtitle}>Submit documents for high-resolution forensic auditing.</p>
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Image Path / Reference Name</label>
            <div style={styles.inputGroup}>
              <input 
                type="text" 
                value={imagePath} 
                onChange={e => setImagePath(e.target.value)} 
                style={styles.input}
                placeholder="e.g. data/example.png"
              />
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', margin: '8px 0' }}>
              <div style={{ flex: 1, height: '1px', backgroundColor: '#334155' }}></div>
              <span style={{ fontSize: '11px', color: '#64748b', padding: '0 8px', fontWeight: 'bold' }}>OR</span>
              <div style={{ flex: 1, height: '1px', backgroundColor: '#334155' }}></div>
            </div>

            <label style={styles.btnSecondary} className="glow-btn">
              {uploading ? 'Uploading...' : 'Upload Image File'}
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleFileUpload} 
                disabled={uploading}
                style={{ display: 'none' }} 
              />
            </label>
          </div>

          {/* Agent Workflow Map */}
          <div style={styles.workflowContainer}>
            <h3 style={styles.sectionTitle}>Agent Workflow Pipeline</h3>
            
            {/* Step 1: Triage */}
            <div style={{...styles.workflowStep, borderColor: activeStep === 'triage' ? '#60a5fa' : '#1e293b'}}>
              <div style={styles.stepHeader}>
                <div style={{...styles.stepNumber, backgroundColor: triageData ? '#10b981' : '#1e293b'}}>1</div>
                <div>
                  <div style={styles.stepTitle}>Triage Agent (Deterministic)</div>
                  <div style={styles.stepDesc}>Runs Exif, pHash, math checks. Zero VLM cost.</div>
                </div>
              </div>
              {triageData && (
                <div style={styles.stepOutput}>
                  Decision: <span style={{color: triageData.decision === 'escalate' ? '#f59e0b' : '#10b981', fontWeight:'bold'}}>{triageData.decision?.toUpperCase()}</span>
                  <span style={{marginLeft: 10, color: '#94a3b8'}}>(Reason: {triageData.reason})</span>
                </div>
              )}
            </div>

            {/* Step 2: Prober */}
            <div style={{...styles.workflowStep, borderColor: activeStep === 'prober' ? '#60a5fa' : '#1e293b'}}>
              <div style={styles.stepHeader}>
                <div style={{...styles.stepNumber, backgroundColor: probeData ? '#10b981' : '#1e293b'}}>2</div>
                <div>
                  <div style={styles.stepTitle}>Prober Agent (Vision / Bbox)</div>
                  <div style={styles.stepDesc}>Locates suspicious text fields & saves high-res crops.</div>
                </div>
              </div>
              {probeData && (
                <div style={styles.stepOutput}>
                  Generated <span style={{color:'#60a5fa', fontWeight:'bold'}}>{probeData.created_cards?.length || 0} Evidence Cards</span>
                </div>
              )}
            </div>

            {/* Step 3: Adjudicator */}
            <div style={{...styles.workflowStep, borderColor: activeStep === 'adjudicator' ? '#60a5fa' : '#1e293b'}}>
              <div style={styles.stepHeader}>
                <div style={{...styles.stepNumber, backgroundColor: adjudicationData ? '#10b981' : '#1e293b'}}>3</div>
                <div>
                  <div style={styles.stepTitle}>Adjudicator Agent (Starved)</div>
                  <div style={styles.stepDesc}>BLIND context: Assesses text-only manifests to formulate claims.</div>
                </div>
              </div>
              {adjudicationData && (
                <div style={styles.stepOutput}>
                  Verdict: <span style={{color: adjudicationData.verdict === 'tamper_detected' ? '#f43f5e' : '#10b981', fontWeight:'bold'}}>{adjudicationData.verdict?.toUpperCase()}</span>
                  <span style={{marginLeft: 10, color: '#94a3b8'}}>({adjudicationData.claims?.length || 0} claims)</span>
                </div>
              )}
            </div>

            {/* Step 4: Verifier */}
            <div style={{...styles.workflowStep, borderColor: activeStep === 'verifier' ? '#60a5fa' : '#1e293b'}}>
              <div style={styles.stepHeader}>
                <div style={{...styles.stepNumber, backgroundColor: verificationData ? '#10b981' : '#1e293b'}}>4</div>
                <div>
                  <div style={styles.stepTitle}>Verifier Agent (Skeptic)</div>
                  <div style={styles.stepDesc}>Context-free review. Checks claims against crops.</div>
                </div>
              </div>
              {verificationData && (
                <div style={styles.stepOutput}>
                  Bonds check: <span style={{color: '#10b981'}}>{verificationData.claims?.filter(c => c.overall === 'holds').length} Holds</span> / <span style={{color: '#f43f5e'}}>{verificationData.claims?.filter(c => c.overall === 'broken').length} Broken</span>
                </div>
              )}
            </div>
          </div>

          {/* Control Actions */}
          <div style={styles.actionButtons}>
            <button 
              onClick={runTriage} 
              disabled={activeStep !== 'idle'} 
              className="glow-btn"
              style={{...styles.btn, backgroundColor: '#3b82f6', opacity: activeStep !== 'idle' ? 0.5 : 1}}
            >
              1. Run Triage
            </button>
            <button 
              onClick={runProbes} 
              disabled={!triageData || activeStep === 'done'} 
              className="glow-btn"
              style={{...styles.btn, backgroundColor: '#8b5cf6', opacity: (!triageData || activeStep === 'done') ? 0.5 : 1}}
            >
              2. Run Prober
            </button>
            <button 
              onClick={runAdjudication} 
              disabled={!probeData || activeStep === 'done'} 
              className="glow-btn"
              style={{...styles.btn, backgroundColor: '#ec4899', opacity: (!probeData || activeStep === 'done') ? 0.5 : 1}}
            >
              3. Adjudicate
            </button>
            <button 
              onClick={runVerification} 
              disabled={!adjudicationData || activeStep === 'done'} 
              className="glow-btn"
              style={{...styles.btn, backgroundColor: '#10b981', opacity: (!adjudicationData || activeStep === 'done') ? 0.5 : 1}}
            >
              4. Verify Bonds
            </button>
          </div>

          <div style={{marginTop: 15, display: 'flex', gap: 10}}>
            <button onClick={clearWorkflow} style={styles.btnSecondary}>Audit New Receipt / Reset</button>
            <button onClick={fetchManifest} style={styles.btnSecondary}>Sync Manifest</button>
          </div>
        </section>

        {/* Center/Right Panel: Evidence & Visualization */}
        <section style={styles.visualizerPanel}>
          
          {/* Tabs/Section: Evidence Ledger Cards */}
          <div style={styles.panelBlock}>
            <div style={styles.cardHeader}>
              <h2 style={styles.cardTitle}>Evidence Ledger ({cards.length} Cards Generated)</h2>
              <p style={styles.cardSubtitle}>Lossless, content-addressed crops containing the raw audit target pixels.</p>
            </div>

            {cards.length === 0 ? (
              <div style={styles.emptyState}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="1.5">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
                <div style={{marginTop: 12, color: '#64748b'}}>No evidence cards created yet. Run the prober agent.</div>
              </div>
            ) : (
              <div style={styles.cardsGrid}>
                {cards.map(card => {
                  const isSelected = selectedCard?.id === card.id;
                  return (
                    <div 
                      key={card.id} 
                      onClick={() => setSelectedCard(card)}
                      style={{
                        ...styles.cardThumbnail,
                        borderColor: isSelected ? '#3b82f6' : '#1e293b',
                        backgroundColor: isSelected ? '#172554' : '#111827'
                      }}
                    >
                      <div style={styles.cardBadge}>{card.probe.toUpperCase()}</div>
                      {card.crop_path ? (
                        <img src={card.crop_path} alt={card.id} style={styles.cropImg} />
                      ) : (
                        <div style={styles.noCropPlaceholder}>No Image Crop</div>
                      )}
                      <div style={styles.thumbnailMeta}>
                        <span style={styles.thumbnailId}>{card.id}</span>
                        <span style={styles.thumbnailSha}>{card.crop_sha256.substring(0, 8)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Selected Evidence Card Inspector */}
          {selectedCard && (
            <div style={styles.inspectorBlock}>
              <div style={styles.inspectorHeader}>
                <h3 style={styles.inspectorTitle}>Card Details: {selectedCard.id}</h3>
                <div style={styles.shaHash}>SHA-256: <span style={{fontFamily: 'JetBrains Mono', color: '#60a5fa'}}>{selectedCard.crop_sha256}</span></div>
              </div>
              <div style={styles.inspectorBody}>
                <div style={styles.inspectorImageContainer}>
                  {selectedCard.crop_path ? (
                    <img src={selectedCard.crop_path} alt="selected crop" style={styles.inspectorImg} />
                  ) : (
                    <div style={styles.noCropPlaceholder}>No Crop Loaded</div>
                  )}
                </div>
                <div style={styles.inspectorMeta}>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Probe Class</span>
                    <span style={styles.metaVal}>{selectedCard.probe}</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Resolution Target</span>
                    <span style={styles.metaVal}>{selectedCard.px_on_target} total pixels</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Verification Cost</span>
                    <span style={styles.metaVal}>{selectedCard.cost_units} units</span>
                  </div>
                  <div style={styles.metaRow}>
                    <span style={styles.metaLabel}>Bounding Box (x1, y1, x2, y2)</span>
                    <span style={styles.metaVal}>{selectedCard.bbox ? JSON.stringify(selectedCard.bbox) : 'Full Frame'}</span>
                  </div>
                  <div style={{...styles.metaRow, flexDirection: 'column', alignItems: 'flex-start', border: 'none'}}>
                    <span style={styles.metaLabel}>Raw Observation Summary</span>
                    <div style={styles.observationBox}>{selectedCard.observation}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Adjudication Claims & Verification status */}
          <div style={styles.panelBlock}>
            <div style={styles.cardHeader}>
              <h2 style={styles.cardTitle}>Adjudication Claims & Evidence Bonds</h2>
              <p style={styles.cardSubtitle}>Formal claims issued by the blindfolded Adjudicator, with verification checks.</p>
            </div>

            {!adjudicationData ? (
              <div style={styles.emptyState}>
                <div style={{color: '#64748b'}}>No claims. Trigger Adjudication to formulate claims based on the manifest.</div>
              </div>
            ) : (
              <div style={styles.claimsList}>
                {adjudicationData.claims?.length === 0 ? (
                  <div style={{padding: 20, textAlign: 'center', color: '#10b981', fontWeight: 'bold'}}>
                    ✓ Clean Audit: Adjudicator found no discrepancies or anomalies.
                  </div>
                ) : (
                  adjudicationData.claims?.map(claim => {
                    // Check if verifier verification has run and has a status for this claim
                    const verificationMatch = verificationData?.claims?.find(c => c.id === claim.id);
                    const bondStatus = verificationMatch ? verificationMatch.overall : 'untested';
                    
                    return (
                      <div key={claim.id} className="claim-card" style={styles.claimCard}>
                        <div style={styles.claimLeft}>
                          <div style={styles.claimText}>{claim.text}</div>
                          <div style={styles.claimBondsList}>
                            <span style={{color: '#64748b'}}>Bonds attached:</span>
                            {claim.bond.map(bId => (
                              <span key={bId} style={styles.bondTag} onClick={() => {
                                const cardMatch = cards.find(c => c.id === bId);
                                if (cardMatch) setSelectedCard(cardMatch);
                              }}>{bId}</span>
                            ))}
                          </div>
                        </div>
                        <div style={styles.claimRight}>
                          <div style={{
                            ...styles.bondBadge,
                            backgroundColor: bondStatus === 'holds' ? '#064e3b' : bondStatus === 'broken' ? '#991b1b' : '#1e293b',
                            color: bondStatus === 'holds' ? '#34d399' : bondStatus === 'broken' ? '#f87171' : '#94a3b8'
                          }}>
                            {bondStatus === 'holds' ? 'BOND HOLDS ✓' : bondStatus === 'broken' ? 'BOND BROKEN ✗' : 'UNTESTED'}
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            )}
          </div>

          {/* Console / System Logs */}
          <div style={styles.panelBlock}>
            <div style={styles.cardHeader}>
              <h2 style={styles.cardTitle}>System Diagnostics Terminal</h2>
              <p style={styles.cardSubtitle}>Real-time monitoring logs from independent agent interactions.</p>
            </div>
            <div style={styles.logTerminal}>
              {logs.length === 0 ? (
                <div style={{color: '#475569', fontStyle: 'italic', fontFamily: 'JetBrains Mono'}}>Terminal initialized. Awaiting pipeline actions...</div>
              ) : (
                logs.map((log, idx) => (
                  <div key={idx} style={styles.logLine}>
                    <span style={styles.logTime}>[{log.time}]</span>{' '}
                    <span style={{
                      color: log.type === 'error' ? '#f43f5e' : 
                             log.type === 'success' ? '#10b981' : 
                             log.type === 'warning' ? '#f59e0b' : 
                             log.type === 'triage' ? '#60a5fa' :
                             log.type === 'prober' ? '#8b5cf6' :
                             log.type === 'adjudicator' ? '#ec4899' : '#e2e8f0'
                    }}>
                      {log.message}
                    </span>
                  </div>
                ))
              )}
            </div>
          </div>

        </section>
      </main>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
    padding: '0 20px 40px 20px',
  },
  header: {
    height: '64px',
    borderBottom: '1px solid #1a222f',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  logoGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  logoText: {
    fontSize: '20px',
    fontWeight: '700',
    letterSpacing: '1px',
    color: '#f8fafc',
  },
  logoBadge: {
    fontSize: '11px',
    fontWeight: '600',
    backgroundColor: '#1e293b',
    color: '#60a5fa',
    padding: '2px 8px',
    borderRadius: '12px',
    textTransform: 'uppercase',
  },
  systemStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  mainGrid: {
    display: 'grid',
    gridTemplateColumns: '380px 1fr',
    gap: '24px',
    alignItems: 'start',
  },
  controlPanel: {
    backgroundColor: '#111827',
    border: '1px solid #1e293b',
    borderRadius: '12px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  cardHeader: {
    marginBottom: '10px',
  },
  cardTitle: {
    fontSize: '18px',
    fontWeight: '600',
    margin: '0 0 4px 0',
    color: '#f1f5f9',
  },
  cardSubtitle: {
    fontSize: '12.5px',
    color: '#64748b',
    margin: 0,
  },
  formGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  label: {
    fontSize: '13px',
    fontWeight: '600',
    color: '#94a3b8',
  },
  inputGroup: {
    display: 'flex',
  },
  input: {
    flex: 1,
    backgroundColor: '#030712',
    border: '1px solid #1e293b',
    borderRadius: '6px',
    color: '#f1f5f9',
    padding: '10px 12px',
    fontSize: '14px',
    fontFamily: 'JetBrains Mono, monospace',
    outline: 'none',
  },
  workflowContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  sectionTitle: {
    fontSize: '14px',
    fontWeight: '700',
    color: '#f1f5f9',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    margin: '0 0 4px 0',
  },
  workflowStep: {
    border: '1px solid #1e293b',
    borderRadius: '8px',
    padding: '12px',
    backgroundColor: '#090d16',
    transition: 'border-color 0.3s ease',
  },
  stepHeader: {
    display: 'flex',
    gap: '12px',
    alignItems: 'center',
  },
  stepNumber: {
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    fontWeight: '700',
    color: '#fff',
    transition: 'background-color 0.3s ease',
  },
  stepTitle: {
    fontSize: '13.5px',
    fontWeight: '600',
    color: '#f1f5f9',
  },
  stepDesc: {
    fontSize: '11px',
    color: '#64748b',
  },
  stepOutput: {
    marginTop: '10px',
    paddingTop: '8px',
    borderTop: '1px dashed #1e293b',
    fontSize: '12px',
    fontFamily: 'JetBrains Mono, monospace',
    color: '#cbd5e1',
  },
  actionButtons: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '10px',
    marginTop: '10px',
  },
  btn: {
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    padding: '12px',
    fontSize: '13px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  btnSecondary: {
    flex: 1,
    backgroundColor: '#1e293b',
    border: '1px solid #334155',
    color: '#cbd5e1',
    borderRadius: '6px',
    padding: '8px 12px',
    fontSize: '12px',
    fontWeight: '600',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    textAlign: 'center',
  },
  visualizerPanel: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px',
  },
  panelBlock: {
    backgroundColor: '#111827',
    border: '1px solid #1e293b',
    borderRadius: '12px',
    padding: '24px',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px',
    border: '2px dashed #1e293b',
    borderRadius: '8px',
  },
  cardsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
    gap: '16px',
    marginTop: '16px',
  },
  cardThumbnail: {
    border: '1px solid #1e293b',
    borderRadius: '8px',
    padding: '10px',
    cursor: 'pointer',
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
    transition: 'all 0.2s ease',
  },
  cardBadge: {
    position: 'absolute',
    top: '6px',
    left: '6px',
    fontSize: '8px',
    fontWeight: '700',
    backgroundColor: '#3b82f6',
    color: '#fff',
    padding: '2px 6px',
    borderRadius: '4px',
    zIndex: 1,
  },
  cropImg: {
    width: '100%',
    aspectRatio: '1',
    objectFit: 'contain',
    borderRadius: '4px',
    backgroundColor: '#000',
  },
  noCropPlaceholder: {
    width: '100%',
    aspectRatio: '1',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#030712',
    color: '#475569',
    fontSize: '11px',
    textAlign: 'center',
    borderRadius: '4px',
  },
  thumbnailMeta: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '10px',
    fontFamily: 'JetBrains Mono, monospace',
    color: '#94a3b8',
  },
  thumbnailId: {
    fontWeight: '700',
  },
  thumbnailSha: {
    color: '#64748b',
  },
  inspectorBlock: {
    backgroundColor: '#1e293b',
    border: '1px solid #3b82f6',
    borderRadius: '12px',
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  inspectorHeader: {
    borderBottom: '1px solid #334155',
    paddingBottom: '12px',
  },
  inspectorTitle: {
    fontSize: '16px',
    fontWeight: '700',
    margin: '0 0 4px 0',
    color: '#f8fafc',
  },
  shaHash: {
    fontSize: '12px',
    color: '#94a3b8',
  },
  inspectorBody: {
    display: 'grid',
    gridTemplateColumns: '200px 1fr',
    gap: '24px',
    alignItems: 'start',
  },
  inspectorImageContainer: {
    width: '100%',
    aspectRatio: '1',
    backgroundColor: '#030712',
    borderRadius: '8px',
    overflow: 'hidden',
    border: '1px solid #334155',
  },
  inspectorImg: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
  },
  inspectorMeta: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  metaRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '8px',
    borderBottom: '1px dashed #334155',
  },
  metaLabel: {
    fontSize: '13px',
    color: '#94a3b8',
    fontWeight: '500',
  },
  metaVal: {
    fontSize: '13.5px',
    color: '#f1f5f9',
    fontWeight: '600',
    fontFamily: 'JetBrains Mono, monospace',
  },
  observationBox: {
    marginTop: '6px',
    width: '100%',
    backgroundColor: '#0f172a',
    border: '1px solid #334155',
    borderRadius: '6px',
    padding: '10px',
    fontSize: '12.5px',
    fontFamily: 'JetBrains Mono, monospace',
    color: '#cbd5e1',
    wordBreak: 'break-all',
  },
  claimsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    marginTop: '16px',
  },
  claimCard: {
    borderRadius: '8px',
    backgroundColor: '#1f2937',
    padding: '16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '16px',
  },
  claimLeft: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  claimText: {
    fontSize: '14.5px',
    fontWeight: '600',
    color: '#f1f5f9',
  },
  claimBondsList: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
  },
  bondTag: {
    backgroundColor: '#374151',
    color: '#60a5fa',
    padding: '2px 8px',
    borderRadius: '4px',
    fontWeight: '600',
    cursor: 'pointer',
  },
  claimRight: {
    display: 'flex',
    alignItems: 'center',
  },
  bondBadge: {
    fontSize: '11px',
    fontWeight: '700',
    padding: '6px 12px',
    borderRadius: '12px',
    whiteSpace: 'nowrap',
    letterSpacing: '0.5px',
  },
  logTerminal: {
    height: '200px',
    overflowY: 'auto',
    backgroundColor: '#030712',
    border: '1px solid #1e293b',
    borderRadius: '8px',
    padding: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  logLine: {
    fontFamily: 'JetBrains Mono, monospace',
    fontSize: '12.5px',
    lineHeight: '1.4',
  },
  logTime: {
    color: '#475569',
  }
};

export default App;
