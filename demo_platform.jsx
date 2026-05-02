import { useState, useEffect, useCallback, useRef } from "react";

// ─── Synthetic Data Engine (mirrors src/data/generator.py) ─────────────
const CATEGORIES = [
  "Savings Account","Checking Account","Credit Card","Personal Loan",
  "Mortgage","Auto Loan","Investment Fund","Retirement Plan (401k/IRA)",
  "Certificate of Deposit (CD)","Money Market Account","Life Insurance",
  "Health Insurance","Brokerage Account","ETF Portfolio","Treasury Bonds",
  "Corporate Bonds","REITs","Annuity","Student Loan Refinance","Home Equity Line (HELOC)"
];
const RISK_LEVELS = ["Conservative","Moderate","Aggressive"];
const RISK_COLOR = { Conservative:"#22c55e", Moderate:"#f59e0b", Aggressive:"#ef4444" };
const AGE_GROUPS = ["18-25","26-35","36-45","46-55","56-65","65+"];
const INCOME_BRACKETS = ["Under $30K","$30K-$50K","$50K-$75K","$75K-$100K","$100K-$150K","$150K-$250K","Over $250K"];
const REGIONS = ["Northeast","Southeast","Midwest","Southwest","West Coast","Pacific NW"];
const CATEGORY_ICONS = {
  "Savings Account":"$","Checking Account":"B","Credit Card":"C","Personal Loan":"L",
  "Mortgage":"M","Auto Loan":"A","Investment Fund":"I","Retirement Plan (401k/IRA)":"R",
  "Certificate of Deposit (CD)":"D","Money Market Account":"MM","Life Insurance":"LI",
  "Health Insurance":"HI","Brokerage Account":"BR","ETF Portfolio":"ET","Treasury Bonds":"TB",
  "Corporate Bonds":"CB","REITs":"RE","Annuity":"AN","Student Loan Refinance":"SL","Home Equity Line (HELOC)":"HE"
};
const RISK_MAP = {
  "Savings Account":"Conservative","Checking Account":"Conservative","Credit Card":"Moderate",
  "Personal Loan":"Moderate","Mortgage":"Moderate","Auto Loan":"Moderate",
  "Investment Fund":"Aggressive","Retirement Plan (401k/IRA)":"Moderate",
  "Certificate of Deposit (CD)":"Conservative","Money Market Account":"Conservative",
  "Life Insurance":"Conservative","Health Insurance":"Conservative",
  "Brokerage Account":"Aggressive","ETF Portfolio":"Moderate","Treasury Bonds":"Conservative",
  "Corporate Bonds":"Moderate","REITs":"Aggressive","Annuity":"Moderate",
  "Student Loan Refinance":"Moderate","Home Equity Line (HELOC)":"Moderate"
};
const APR_RANGES = {
  "Savings Account":[0.5,5],"Checking Account":[0,0.5],"Credit Card":[15,28],"Personal Loan":[6,18],
  "Mortgage":[3,8],"Auto Loan":[3.5,12],"Investment Fund":[4,15],"Retirement Plan (401k/IRA)":[5,12],
  "Certificate of Deposit (CD)":[2,5.5],"Money Market Account":[1.5,5],"Life Insurance":[2,6],
  "Health Insurance":[0,0],"Brokerage Account":[5,20],"ETF Portfolio":[6,15],"Treasury Bonds":[2,5.5],
  "Corporate Bonds":[3.5,8],"REITs":[5,14],"Annuity":[3,7],"Student Loan Refinance":[3.5,10],"Home Equity Line (HELOC)":[4,12]
};

const rand = (a,b) => Math.random()*(b-a)+a;
const pick = arr => arr[Math.floor(Math.random()*arr.length)];
const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
const gaussRand = (m,s) => { let u=0,v=0; while(!u) u=Math.random(); while(!v) v=Math.random(); return m+s*Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

function generateUser(id) {
  return {
    user_id: id,
    name: `User ${id.toString().padStart(4,"0")}`,
    age_group: pick(AGE_GROUPS),
    income_bracket: pick(INCOME_BRACKETS),
    region: pick(REGIONS),
    credit_score: clamp(Math.round(gaussRand(700,80)),300,850),
    risk_tolerance: pick(RISK_LEVELS),
    years_as_customer: Math.min(30, Math.max(0, Math.round(Math.random()*15))),
    num_existing_products: Math.min(12, Math.max(0, Math.round(Math.random()*5))),
    digital_engagement: Math.round(rand(5,95)),
    total_assets: Math.round(rand(5000,500000)),
  };
}

function generateProduct(id) {
  const cat = CATEGORIES[id % CATEGORIES.length];
  const [lo,hi] = APR_RANGES[cat];
  return {
    product_id: id,
    product_name: `${cat} - Plan ${String.fromCharCode(65 + id%26)}${Math.floor(id/26)+1}`,
    category: cat,
    risk_level: RISK_MAP[cat],
    annual_return_pct: Math.round(rand(lo,hi)*100)/100,
    fee_pct: Math.round(rand(0,2.5)*100)/100,
    term_months: pick([0,6,12,24,36,60,120,360]),
    is_tax_advantaged: ["Retirement Plan (401k/IRA)","Treasury Bonds","Life Insurance","Health Insurance"].includes(cat),
    popularity: Math.round(rand(10,100)),
  };
}

function generateRecommendations(user, products, topK=8) {
  const scored = products.map(p => {
    let score = 0.3 + Math.random()*0.5;
    if (p.risk_level === user.risk_tolerance) score += 0.15;
    if (user.credit_score > 700 && ["Mortgage","Investment Fund","Brokerage Account"].includes(p.category)) score += 0.1;
    if (user.income_bracket.includes("150K") || user.income_bracket.includes("250K")) {
      if (["Investment Fund","ETF Portfolio","REITs","Brokerage Account"].includes(p.category)) score += 0.12;
    }
    if (user.age_group === "56-65" || user.age_group === "65+") {
      if (["Retirement Plan (401k/IRA)","Annuity","Treasury Bonds","Life Insurance"].includes(p.category)) score += 0.15;
    }
    if (user.age_group === "18-25" || user.age_group === "26-35") {
      if (["Credit Card","Savings Account","Student Loan Refinance"].includes(p.category)) score += 0.1;
    }
    score = clamp(score + (p.popularity/100)*0.1, 0, 0.99);
    return { ...p, recommendation_score: Math.round(score*1000)/1000, confidence: score>=0.75?"Very High":score>=0.6?"High":score>=0.45?"Medium":"Low" };
  });
  scored.sort((a,b) => b.recommendation_score - a.recommendation_score);
  return scored.slice(0, topK);
}

function generateTrainingHistory() {
  const epochs = 25;
  const trainLoss = []; const valLoss = []; const valAuc = [];
  let tl = 0.68; let vl = 0.72; let auc = 0.52;
  for (let i=0; i<epochs; i++) {
    tl = Math.max(0.15, tl - rand(0.01,0.04));
    vl = Math.max(0.28, vl - rand(0.005,0.03));
    auc = Math.min(0.92, auc + rand(0.005,0.025));
    trainLoss.push(Math.round(tl*1000)/1000);
    valLoss.push(Math.round(vl*1000)/1000);
    valAuc.push(Math.round(auc*1000)/1000);
  }
  return { trainLoss, valLoss, valAuc, epochs };
}

// ─── Main App ──────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [users] = useState(() => Array.from({length:50}, (_,i) => generateUser(i)));
  const [products] = useState(() => Array.from({length:60}, (_,i) => generateProduct(i)));
  const [selectedUser, setSelectedUser] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [riskFilter, setRiskFilter] = useState("All");
  const [categoryFilter, setCategoryFilter] = useState("All");
  const [topK, setTopK] = useState(8);
  const [trainingData] = useState(generateTrainingHistory);
  const [isLoading, setIsLoading] = useState(false);
  const [animatedCards, setAnimatedCards] = useState(new Set());

  useEffect(() => { setSelectedUser(users[0]); }, [users]);
  
  useEffect(() => {
    if (!selectedUser) return;
    setIsLoading(true);
    setAnimatedCards(new Set());
    const timer = setTimeout(() => {
      let filtered = products;
      if (riskFilter !== "All") filtered = filtered.filter(p => p.risk_level === riskFilter);
      if (categoryFilter !== "All") filtered = filtered.filter(p => p.category === categoryFilter);
      const recs = generateRecommendations(selectedUser, filtered, topK);
      setRecommendations(recs);
      setIsLoading(false);
      recs.forEach((_,i) => setTimeout(() => setAnimatedCards(prev => new Set([...prev, i])), i*80));
    }, 600);
    return () => clearTimeout(timer);
  }, [selectedUser, riskFilter, categoryFilter, topK, products]);

  const metrics = {
    "HR@10": "0.78", "nDCG@10": "0.62", "Precision@10": "0.34",
    "Recall@10": "0.51", "MRR": "0.58", "AUC-ROC": trainingData.valAuc[trainingData.valAuc.length-1].toFixed(3),
    "Diversity": "0.72", "Coverage": "0.85"
  };

  const tabs = [
    { id:"dashboard", label:"Dashboard", icon:"◉" },
    { id:"recommend", label:"Recommendations", icon:"" },
    { id:"training", label:"Model Training", icon:"" },
    { id:"architecture", label:"Architecture", icon:"◫" },
  ];

  return (
    <div style={styles.root}>
      {/* ── Sidebar ── */}
      <nav style={styles.sidebar}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>◈</span>
          <div>
            <div style={styles.logoTitle}>FinRecSys</div>
            <div style={styles.logoSub}>NeuMF Engine v1.0</div>
          </div>
        </div>
        <div style={styles.navGroup}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)}
              style={{...styles.navBtn, ...(activeTab===t.id ? styles.navBtnActive : {})}}>
              <span style={styles.navIcon}>{t.icon}</span> {t.label}
            </button>
          ))}
        </div>
        <div style={styles.sidebarFooter}>
          <div style={styles.footerBadge}>PyTorch + FastAPI</div>
          <div style={{fontSize:11,color:"#64748b",marginTop:6}}>22 Tests Passing </div>
        </div>
      </nav>

      {/* ── Main Content ── */}
      <main style={styles.main}>
        {activeTab === "dashboard" && <DashboardView users={users} products={products} metrics={metrics} trainingData={trainingData} onNavigate={setActiveTab} />}
        {activeTab === "recommend" && (
          <RecommendView
            users={users} selectedUser={selectedUser} setSelectedUser={setSelectedUser}
            recommendations={recommendations} isLoading={isLoading} animatedCards={animatedCards}
            riskFilter={riskFilter} setRiskFilter={setRiskFilter}
            categoryFilter={categoryFilter} setCategoryFilter={setCategoryFilter}
            topK={topK} setTopK={setTopK} products={products}
          />
        )}
        {activeTab === "training" && <TrainingView data={trainingData} metrics={metrics} />}
        {activeTab === "architecture" && <ArchitectureView />}
      </main>
    </div>
  );
}

// ─── Dashboard View ────────────────────────────────────────────────
function DashboardView({ users, products, metrics, trainingData, onNavigate }) {
  const statCards = [
    { label:"Users", value: users.length.toLocaleString(), delta:"+12%", color:"#0ea5e9" },
    { label:"Products", value: products.length, delta:"20 categories", color:"#8b5cf6" },
    { label:"AUC-ROC", value: metrics["AUC-ROC"], delta:"Test Set", color:"#22c55e" },
    { label:"nDCG@10", value: metrics["nDCG@10"], delta:"Ranking Quality", color:"#f59e0b" },
  ];
  const riskDist = RISK_LEVELS.map(r => ({ risk: r, count: products.filter(p => p.risk_level===r).length }));
  const topCategories = {};
  products.forEach(p => { topCategories[p.category] = (topCategories[p.category]||0)+1; });
  const sortedCats = Object.entries(topCategories).sort((a,b)=>b[1]-a[1]).slice(0,8);
  const maxCatCount = Math.max(...sortedCats.map(c=>c[1]));

  return (
    <div style={styles.page}>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>System Overview</h1>
        <p style={styles.pageSubtitle}>Deep Learning Recommendation System for Financial Services</p>
      </div>
      
      <div style={styles.statGrid}>
        {statCards.map((s,i) => (
          <div key={i} style={{...styles.statCard, animationDelay:`${i*0.1}s`}}>
            <div style={{...styles.statDot, background:s.color}} />
            <div style={styles.statValue}>{s.value}</div>
            <div style={styles.statLabel}>{s.label}</div>
            <div style={styles.statDelta}>{s.delta}</div>
          </div>
        ))}
      </div>

      <div style={styles.twoCol}>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Model Performance</h3>
          <div style={styles.metricsGrid}>
            {Object.entries(metrics).map(([k,v]) => (
              <div key={k} style={styles.metricRow}>
                <span style={styles.metricLabel}>{k}</span>
                <div style={styles.metricBarOuter}>
                  <div style={{...styles.metricBarInner, width:`${parseFloat(v)*100}%`}} />
                </div>
                <span style={styles.metricValue}>{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Product Distribution</h3>
          <div style={{marginTop:12}}>
            <div style={{display:"flex",gap:16,marginBottom:20}}>
              {riskDist.map(r => (
                <div key={r.risk} style={{flex:1,textAlign:"center",padding:"12px 8px",borderRadius:10,background:"#f8fafc"}}>
                  <div style={{width:10,height:10,borderRadius:"50%",background:RISK_COLOR[r.risk],margin:"0 auto 6px"}} />
                  <div style={{fontSize:20,fontWeight:700,color:"#0f172a"}}>{r.count}</div>
                  <div style={{fontSize:11,color:"#64748b"}}>{r.risk}</div>
                </div>
              ))}
            </div>
            {sortedCats.map(([cat,cnt]) => (
              <div key={cat} style={{display:"flex",alignItems:"center",gap:8,marginBottom:8}}>
                <span style={{fontSize:14,width:20}}>{CATEGORY_ICONS[cat]}</span>
                <span style={{fontSize:12,color:"#475569",width:100,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{cat.split("(")[0].trim()}</span>
                <div style={{flex:1,height:6,background:"#f1f5f9",borderRadius:3,overflow:"hidden"}}>
                  <div style={{height:"100%",width:`${(cnt/maxCatCount)*100}%`,background:"linear-gradient(90deg,#6366f1,#8b5cf6)",borderRadius:3,transition:"width 1s ease"}} />
                </div>
                <span style={{fontSize:12,fontWeight:600,color:"#334155",width:20,textAlign:"right"}}>{cnt}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Training Loss Curve</h3>
        <MiniChart trainLoss={trainingData.trainLoss} valLoss={trainingData.valLoss} valAuc={trainingData.valAuc} />
      </div>

      <button onClick={() => onNavigate("recommend")} style={styles.ctaButton}>
         Try the Recommendation Engine →
      </button>
    </div>
  );
}

// ─── Mini Chart (SVG) ──────────────────────────────────────────────
function MiniChart({ trainLoss, valLoss, valAuc }) {
  const W=680, H=200, pad=40;
  const n = trainLoss.length;
  const maxY = Math.max(...trainLoss,...valLoss)*1.1;
  const x = i => pad + (i/(n-1))*(W-2*pad);
  const y = v => pad + (1 - v/maxY)*(H-2*pad);
  const path = arr => arr.map((v,i) => `${i===0?"M":"L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:200}}>
      <defs>
        <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6366f1" stopOpacity="0.2"/>
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {[0,0.25,0.5,0.75,1].map(f => (
        <g key={f}>
          <line x1={pad} x2={W-pad} y1={y(maxY*f)} y2={y(maxY*f)} stroke="#e2e8f0" strokeWidth="1"/>
          <text x={pad-6} y={y(maxY*f)+4} textAnchor="end" fontSize="10" fill="#94a3b8">{(maxY*f).toFixed(2)}</text>
        </g>
      ))}
      <path d={`${path(trainLoss)} L${x(n-1)},${H-pad} L${x(0)},${H-pad} Z`} fill="url(#tg)"/>
      <path d={path(trainLoss)} fill="none" stroke="#6366f1" strokeWidth="2.5" strokeLinejoin="round"/>
      <path d={path(valLoss)} fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="6,3" strokeLinejoin="round"/>
      <g transform={`translate(${W-pad-160},${pad-10})`}>
        <line x1="0" y1="6" x2="16" y2="6" stroke="#6366f1" strokeWidth="2.5"/>
        <text x="20" y="10" fontSize="11" fill="#475569">Train Loss</text>
        <line x1="85" y1="6" x2="101" y2="6" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4,2"/>
        <text x="105" y="10" fontSize="11" fill="#475569">Val Loss</text>
      </g>
      {trainLoss.map((v,i) => i%5===0 ? <text key={i} x={x(i)} y={H-pad+16} textAnchor="middle" fontSize="10" fill="#94a3b8">{i+1}</text> : null)}
      <text x={W/2} y={H-4} textAnchor="middle" fontSize="11" fill="#94a3b8">Epoch</text>
    </svg>
  );
}

// ─── Recommend View ────────────────────────────────────────────────
function RecommendView({ users, selectedUser, setSelectedUser, recommendations, isLoading, animatedCards, riskFilter, setRiskFilter, categoryFilter, setCategoryFilter, topK, setTopK, products }) {
  const uniqueCats = ["All", ...new Set(products.map(p=>p.category))];
  
  return (
    <div style={styles.page}>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>Recommendation Engine</h1>
        <p style={styles.pageSubtitle}>Neural Collaborative Filtering - Personalized Financial Products</p>
      </div>

      <div style={styles.twoCol}>
        {/* User Selector + Profile */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Select User</h3>
          <select value={selectedUser?.user_id??0} onChange={e => setSelectedUser(users.find(u=>u.user_id===+e.target.value))} style={styles.select}>
            {users.map(u => <option key={u.user_id} value={u.user_id}>User {u.user_id.toString().padStart(4,"0")} - {u.age_group}, {u.income_bracket}</option>)}
          </select>
          {selectedUser && (
            <div style={{marginTop:16}}>
              <div style={styles.profileGrid}>
                <ProfileField label="Age Group" value={selectedUser.age_group} />
                <ProfileField label="Income" value={selectedUser.income_bracket} />
                <ProfileField label="Region" value={selectedUser.region} />
                <ProfileField label="Risk Tolerance" value={selectedUser.risk_tolerance} color={RISK_COLOR[selectedUser.risk_tolerance]} />
                <ProfileField label="Credit Score" value={selectedUser.credit_score} />
                <ProfileField label="Years as Customer" value={selectedUser.years_as_customer} />
                <ProfileField label="Existing Products" value={selectedUser.num_existing_products} />
                <ProfileField label="Digital Engagement" value={`${selectedUser.digital_engagement}%`} />
                <ProfileField label="Total Assets" value={`$${(selectedUser.total_assets/1000).toFixed(0)}K`} />
              </div>
              <div style={{marginTop:12,padding:"10px 14px",background:"#f0fdf4",borderRadius:8,border:"1px solid #bbf7d0"}}>
                <span style={{fontSize:12,color:"#15803d",fontWeight:600}}>Credit Score: </span>
                <span style={{fontSize:14,fontWeight:700,color:selectedUser.credit_score>=700?"#15803d":selectedUser.credit_score>=600?"#d97706":"#dc2626"}}>{selectedUser.credit_score}</span>
                <div style={{height:4,background:"#dcfce7",borderRadius:2,marginTop:6}}>
                  <div style={{height:"100%",width:`${((selectedUser.credit_score-300)/550)*100}%`,background:selectedUser.credit_score>=700?"#22c55e":selectedUser.credit_score>=600?"#f59e0b":"#ef4444",borderRadius:2,transition:"width 0.6s ease"}} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Filters */}
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Filters & Controls</h3>
          <div style={{display:"flex",flexDirection:"column",gap:14,marginTop:8}}>
            <div>
              <label style={styles.filterLabel}>Risk Level</label>
              <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                {["All",...RISK_LEVELS].map(r => (
                  <button key={r} onClick={() => setRiskFilter(r)}
                    style={{...styles.filterChip, ...(riskFilter===r ? styles.filterChipActive : {})}}>
                    {r !== "All" && <span style={{width:8,height:8,borderRadius:"50%",background:RISK_COLOR[r],display:"inline-block",marginRight:4}} />}
                    {r}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label style={styles.filterLabel}>Category</label>
              <select value={categoryFilter} onChange={e=>setCategoryFilter(e.target.value)} style={styles.select}>
                {uniqueCats.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label style={styles.filterLabel}>Top-K Results: {topK}</label>
              <input type="range" min={3} max={15} value={topK} onChange={e=>setTopK(+e.target.value)} style={styles.rangeInput} />
            </div>
          </div>
          <div style={{marginTop:20,padding:"14px",background:"#faf5ff",borderRadius:10,border:"1px solid #e9d5ff"}}>
            <div style={{fontSize:11,fontWeight:600,color:"#7c3aed",textTransform:"uppercase",letterSpacing:1,marginBottom:6}}>Model Info</div>
            <div style={{fontSize:12,color:"#6b21a8",lineHeight:1.7}}>
              NeuMF: GMF(64-d) + MLP[256→128→64→32]<br/>
              Optimizer: Adam (lr=0.001, wd=1e-5)<br/>
              Negative Sampling: 4:1 ratio<br/>
              Scoring: 85% CF + 15% Popularity
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations Grid */}
      <div style={{marginTop:24}}>
        <h3 style={{...styles.cardTitle, marginBottom:16}}>
          {isLoading ? "Computing recommendations..." : `Top ${recommendations.length} Recommendations for User ${selectedUser?.user_id.toString().padStart(4,"0")}`}
        </h3>
        {isLoading ? (
          <div style={{textAlign:"center",padding:60}}>
            <div style={styles.spinner} />
            <div style={{marginTop:16,color:"#64748b",fontSize:13}}>Running NeuMF inference...</div>
          </div>
        ) : (
          <div style={styles.recGrid}>
            {recommendations.map((rec, i) => (
              <div key={rec.product_id} style={{
                ...styles.recCard,
                opacity: animatedCards.has(i) ? 1 : 0,
                transform: animatedCards.has(i) ? "translateY(0)" : "translateY(20px)",
                transition: "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)"
              }}>
                <div style={styles.recHeader}>
                  <span style={{fontSize:24}}>{CATEGORY_ICONS[rec.category]}</span>
                  <span style={{...styles.recBadge, background: rec.confidence==="Very High"?"#dcfce7":rec.confidence==="High"?"#fef9c3":"#f1f5f9",
                    color: rec.confidence==="Very High"?"#15803d":rec.confidence==="High"?"#a16207":"#475569"}}>
                    {rec.confidence}
                  </span>
                </div>
                <div style={styles.recCategory}>{rec.category}</div>
                <div style={styles.recName}>{rec.product_name}</div>
                <div style={styles.recDetails}>
                  <div style={styles.recDetail}>
                    <span style={styles.recDetailLabel}>Score</span>
                    <span style={styles.recDetailValue}>{rec.recommendation_score.toFixed(3)}</span>
                  </div>
                  <div style={styles.recDetail}>
                    <span style={styles.recDetailLabel}>Return</span>
                    <span style={{...styles.recDetailValue, color:rec.annual_return_pct>0?"#15803d":"#64748b"}}>{rec.annual_return_pct.toFixed(1)}%</span>
                  </div>
                  <div style={styles.recDetail}>
                    <span style={styles.recDetailLabel}>Fee</span>
                    <span style={styles.recDetailValue}>{rec.fee_pct.toFixed(2)}%</span>
                  </div>
                </div>
                <div style={{display:"flex",alignItems:"center",gap:6,marginTop:10}}>
                  <span style={{width:8,height:8,borderRadius:"50%",background:RISK_COLOR[rec.risk_level]}} />
                  <span style={{fontSize:11,color:"#64748b"}}>{rec.risk_level}</span>
                  {rec.is_tax_advantaged && <span style={{fontSize:10,background:"#eff6ff",color:"#2563eb",padding:"2px 6px",borderRadius:4,marginLeft:"auto"}}>Tax Advantaged</span>}
                </div>
                <div style={{height:3,background:"#f1f5f9",borderRadius:2,marginTop:10,overflow:"hidden"}}>
                  <div style={{height:"100%",width:`${rec.recommendation_score*100}%`,background:`linear-gradient(90deg, #6366f1, ${RISK_COLOR[rec.risk_level]})`,borderRadius:2,transition:"width 0.8s ease"}} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProfileField({ label, value, color }) {
  return (
    <div style={{padding:"8px 10px",background:"#f8fafc",borderRadius:8}}>
      <div style={{fontSize:10,color:"#94a3b8",textTransform:"uppercase",letterSpacing:0.5}}>{label}</div>
      <div style={{fontSize:13,fontWeight:600,color:color||"#1e293b",marginTop:2}}>{value}</div>
    </div>
  );
}

// ─── Training View ─────────────────────────────────────────────────
function TrainingView({ data, metrics }) {
  return (
    <div style={styles.page}>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>Model Training & Evaluation</h1>
        <p style={styles.pageSubtitle}>NeuMF training convergence and recommendation quality metrics</p>
      </div>
      
      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Loss Convergence</h3>
        <MiniChart trainLoss={data.trainLoss} valLoss={data.valLoss} valAuc={data.valAuc} />
        <div style={{display:"flex",justifyContent:"space-around",marginTop:16,padding:"12px 0",borderTop:"1px solid #f1f5f9"}}>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:20,fontWeight:700,color:"#6366f1"}}>{data.trainLoss[data.trainLoss.length-1]}</div>
            <div style={{fontSize:11,color:"#94a3b8"}}>Final Train Loss</div>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:20,fontWeight:700,color:"#f59e0b"}}>{data.valLoss[data.valLoss.length-1]}</div>
            <div style={{fontSize:11,color:"#94a3b8"}}>Final Val Loss</div>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:20,fontWeight:700,color:"#22c55e"}}>{data.valAuc[data.valAuc.length-1]}</div>
            <div style={{fontSize:11,color:"#94a3b8"}}>Best AUC-ROC</div>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:20,fontWeight:700,color:"#0ea5e9"}}>{data.epochs}</div>
            <div style={{fontSize:11,color:"#94a3b8"}}>Epochs</div>
          </div>
        </div>
      </div>

      <div style={styles.twoCol}>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Recommendation Metrics</h3>
          <div style={{display:"flex",flexDirection:"column",gap:10,marginTop:8}}>
            {Object.entries(metrics).map(([k,v]) => (
              <div key={k} style={{display:"flex",alignItems:"center",gap:12}}>
                <span style={{fontSize:12,color:"#64748b",width:90}}>{k}</span>
                <div style={{flex:1,height:8,background:"#f1f5f9",borderRadius:4,overflow:"hidden"}}>
                  <div style={{height:"100%",width:`${parseFloat(v)*100}%`,background:"linear-gradient(90deg,#6366f1,#a78bfa)",borderRadius:4,transition:"width 1.2s cubic-bezier(0.16, 1, 0.3, 1)"}} />
                </div>
                <span style={{fontSize:13,fontWeight:700,color:"#1e293b",width:42,textAlign:"right"}}>{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Training Configuration</h3>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginTop:8}}>
            {[
              ["Architecture","NeuMF (GMF+MLP)"],["Embedding Dim","64"],
              ["MLP Layers","256→128→64→32"],["Dropout","0.2"],
              ["Optimizer","Adam"],["Learning Rate","0.001"],
              ["Batch Size","256"],["Loss Function","Binary CE"],
              ["Neg Samples","4:1 ratio"],["Early Stop","patience=5"],
              ["LR Schedule","ReduceOnPlateau"],["Grad Clip","max_norm=1.0"],
            ].map(([k,v]) => (
              <div key={k} style={{padding:"8px 10px",background:"#f8fafc",borderRadius:8}}>
                <div style={{fontSize:10,color:"#94a3b8",textTransform:"uppercase",letterSpacing:0.5}}>{k}</div>
                <div style={{fontSize:12,fontWeight:600,color:"#1e293b",marginTop:2}}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>AUC-ROC Progression</h3>
        <AucChart data={data.valAuc} />
      </div>
    </div>
  );
}

function AucChart({ data }) {
  const W=680, H=180, pad=40;
  const n=data.length;
  const minY=Math.min(...data)*0.95, maxY=Math.min(1, Math.max(...data)*1.05);
  const x=i=>pad+(i/(n-1))*(W-2*pad);
  const y=v=>pad+(1-(v-minY)/(maxY-minY))*(H-2*pad);
  const pts=data.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{width:"100%",height:180}}>
      <defs>
        <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22c55e" stopOpacity="0.25"/><stop offset="100%" stopColor="#22c55e" stopOpacity="0"/>
        </linearGradient>
      </defs>
      {[0,0.25,0.5,0.75,1].map(f => {
        const val = minY + f*(maxY-minY);
        return <g key={f}><line x1={pad} x2={W-pad} y1={y(val)} y2={y(val)} stroke="#e2e8f0" strokeWidth="1"/>
          <text x={pad-6} y={y(val)+4} textAnchor="end" fontSize="10" fill="#94a3b8">{val.toFixed(2)}</text></g>;
      })}
      <polygon points={`${pts} ${x(n-1)},${H-pad} ${x(0)},${H-pad}`} fill="url(#ag)"/>
      <polyline points={pts} fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinejoin="round"/>
      {data.map((v,i) => <circle key={i} cx={x(i)} cy={y(v)} r="3" fill="#fff" stroke="#22c55e" strokeWidth="2"/>)}
    </svg>
  );
}

// ─── Architecture View ─────────────────────────────────────────────
function ArchitectureView() {
  return (
    <div style={styles.page}>
      <div style={styles.pageHeader}>
        <h1 style={styles.pageTitle}>System Architecture</h1>
        <p style={styles.pageSubtitle}>Neural Collaborative Filtering pipeline - data flow from ingestion to serving</p>
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>NeuMF Model Architecture</h3>
        <ArchDiagram />
      </div>

      <div style={styles.twoCol}>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Data Pipeline</h3>
          <div style={{display:"flex",flexDirection:"column",gap:12,marginTop:8}}>
            {[
              { step:"1", title:"Data Generation", desc:"Synthetic users (credit scores, risk profiles), 20 product categories, implicit feedback signals" },
              { step:"2", title:"Preprocessing", desc:"Label encoding, normalization, temporal train/val/test split, 4:1 negative sampling" },
              { step:"3", title:"Model Training", desc:"NeuMF forward pass → BCE loss → Adam optimizer with gradient clipping and LR scheduling" },
              { step:"4", title:"Inference", desc:"Score all candidate products, blend CF (85%) + popularity (15%), rank and return top-K" },
              { step:"5", title:"API Serving", desc:"FastAPI REST endpoints, CORS-enabled, auto-initialization on startup" },
            ].map(s => (
              <div key={s.step} style={{display:"flex",gap:12,alignItems:"flex-start"}}>
                <div style={{width:28,height:28,borderRadius:"50%",background:"#6366f1",color:"#fff",display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:700,flexShrink:0}}>{s.step}</div>
                <div>
                  <div style={{fontSize:13,fontWeight:600,color:"#1e293b"}}>{s.title}</div>
                  <div style={{fontSize:12,color:"#64748b",marginTop:2,lineHeight:1.5}}>{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div style={styles.card}>
          <h3 style={styles.cardTitle}>Tech Stack</h3>
          <div style={{display:"flex",flexDirection:"column",gap:10,marginTop:8}}>
            {[
              { cat:"Deep Learning", items:"PyTorch 2.0+, NeuMF, GMF, MLP, DeepFM" },
              { cat:"Data", items:"Pandas, NumPy, scikit-learn, Synthetic Generator" },
              { cat:"API", items:"FastAPI, Uvicorn, Pydantic, CORS" },
              { cat:"Metrics", items:"HR@K, nDCG@K, Precision, Recall, MRR, AUC" },
              { cat:"Testing", items:"pytest (22 tests), Unit + Integration" },
              { cat:"Deploy", items:"Docker, Health checks, Model checkpointing" },
            ].map(t => (
              <div key={t.cat} style={{padding:"10px 14px",background:"#f8fafc",borderRadius:8,borderLeft:"3px solid #6366f1"}}>
                <div style={{fontSize:11,fontWeight:700,color:"#6366f1",textTransform:"uppercase",letterSpacing:0.5}}>{t.cat}</div>
                <div style={{fontSize:12,color:"#475569",marginTop:3}}>{t.items}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={styles.card}>
        <h3 style={styles.cardTitle}>Project Structure</h3>
        <pre style={styles.codeBlock}>{`Deep-Learning-Recommendation-System-for-Financial-Services/
├── README.md                    # Documentation + Architecture
├── main.py                      # End-to-end pipeline (generate → train → evaluate)
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container deployment
├── setup.py                     # Package config
│
├── src/
│   ├── config/settings.py       # Centralized hyperparameters
│   ├── data/
│   │   ├── generator.py         # Synthetic financial data (users, products, interactions)
│   │   └── preprocessor.py      # Encoding, splits, negative sampling, DataLoaders
│   ├── models/
│   │   ├── neumf.py             # NeuMF = GMF + MLP + Fusion (+ DeepFM, Hybrid)
│   │   ├── trainer.py           # Training loop, early stopping, checkpointing
│   │   ├── recommender.py       # Top-K engine, similarity search, trending
│   │   └── metrics.py           # HR@K, nDCG@K, Precision, Recall, MRR, Coverage
│   └── api/app.py               # FastAPI (12 endpoints)
│
└── tests/test_system.py         # 22 tests (config, data, model, training, metrics)`}</pre>
      </div>
    </div>
  );
}

function ArchDiagram() {
  return (
    <svg viewBox="0 0 700 420" style={{width:"100%",height:420}}>
      <defs>
        <linearGradient id="grd1" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor="#6366f1"/><stop offset="100%" stopColor="#8b5cf6"/></linearGradient>
        <linearGradient id="grd2" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor="#0ea5e9"/><stop offset="100%" stopColor="#06b6d4"/></linearGradient>
        <linearGradient id="grd3" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stopColor="#22c55e"/><stop offset="100%" stopColor="#10b981"/></linearGradient>
        <filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="3" floodOpacity="0.1"/></filter>
      </defs>
      
      {/* Input Layer */}
      <rect x="80" y="20" width="120" height="40" rx="8" fill="url(#grd2)" filter="url(#sh)"/>
      <text x="140" y="44" textAnchor="middle" fill="#fff" fontSize="12" fontWeight="600">User ID</text>
      <rect x="500" y="20" width="120" height="40" rx="8" fill="url(#grd2)" filter="url(#sh)"/>
      <text x="560" y="44" textAnchor="middle" fill="#fff" fontSize="12" fontWeight="600">Product ID</text>
      
      {/* Arrows down */}
      <line x1="115" y1="60" x2="115" y2="90" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arrow)"/>
      <line x1="165" y1="60" x2="165" y2="90" stroke="#94a3b8" strokeWidth="1.5"/>
      <line x1="535" y1="60" x2="535" y2="90" stroke="#94a3b8" strokeWidth="1.5"/>
      <line x1="585" y1="60" x2="585" y2="90" stroke="#94a3b8" strokeWidth="1.5"/>

      {/* Embedding Layer */}
      <rect x="50" y="90" width="100" height="36" rx="6" fill="#ede9fe" stroke="#8b5cf6" strokeWidth="1.5"/>
      <text x="100" y="112" textAnchor="middle" fill="#6d28d9" fontSize="10" fontWeight="600">GMF Embed (64)</text>
      <rect x="170" y="90" width="100" height="36" rx="6" fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.5"/>
      <text x="220" y="112" textAnchor="middle" fill="#b45309" fontSize="10" fontWeight="600">MLP Embed (64)</text>
      
      <rect x="470" y="90" width="100" height="36" rx="6" fill="#ede9fe" stroke="#8b5cf6" strokeWidth="1.5"/>
      <text x="520" y="112" textAnchor="middle" fill="#6d28d9" fontSize="10" fontWeight="600">GMF Embed (64)</text>
      <rect x="590" y="90" width="100" height="36" rx="6" fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.5"/>
      <text x="640" y="112" textAnchor="middle" fill="#b45309" fontSize="10" fontWeight="600">MLP Embed (64)</text>

      {/* GMF Path */}
      <line x1="100" y1="126" x2="100" y2="170" stroke="#8b5cf6" strokeWidth="1.5"/>
      <line x1="520" y1="126" x2="520" y2="170" stroke="#8b5cf6" strokeWidth="1.5"/>
      <line x1="100" y1="170" x2="200" y2="190" stroke="#8b5cf6" strokeWidth="1.5"/>
      <line x1="520" y1="170" x2="420" y2="190" stroke="#8b5cf6" strokeWidth="1.5"/>
      
      <rect x="220" y="175" width="180" height="36" rx="6" fill="url(#grd1)" filter="url(#sh)"/>
      <text x="310" y="197" textAnchor="middle" fill="#fff" fontSize="11" fontWeight="600">⊙ Element-wise Product (64-d)</text>

      {/* MLP Path */}
      <line x1="220" y1="126" x2="220" y2="155" stroke="#f59e0b" strokeWidth="1.5"/>
      <line x1="640" y1="126" x2="640" y2="155" stroke="#f59e0b" strokeWidth="1.5"/>
      <line x1="220" y1="155" x2="430" y2="155" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4,3"/>
      <line x1="640" y1="155" x2="430" y2="155" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4,3"/>
      
      <rect x="360" y="230" width="140" height="30" rx="5" fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.2"/>
      <text x="430" y="249" textAnchor="middle" fill="#92400e" fontSize="10" fontWeight="600">Concat → Dense(256)</text>
      <rect x="360" y="265" width="140" height="30" rx="5" fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.2"/>
      <text x="430" y="284" textAnchor="middle" fill="#92400e" fontSize="10" fontWeight="600">BN → ReLU → Dense(128)</text>
      <rect x="360" y="300" width="140" height="30" rx="5" fill="#fef3c7" stroke="#f59e0b" strokeWidth="1.2"/>
      <text x="430" y="319" textAnchor="middle" fill="#92400e" fontSize="10" fontWeight="600">BN → ReLU → Dense(32)</text>

      <line x1="430" y1="155" x2="430" y2="230" stroke="#f59e0b" strokeWidth="1.5"/>
      <line x1="430" y1="260" x2="430" y2="265" stroke="#f59e0b" strokeWidth="1"/>
      <line x1="430" y1="295" x2="430" y2="300" stroke="#f59e0b" strokeWidth="1"/>

      {/* Fusion */}
      <line x1="310" y1="211" x2="310" y2="360" stroke="#8b5cf6" strokeWidth="1.5"/>
      <line x1="430" y1="330" x2="430" y2="360" stroke="#f59e0b" strokeWidth="1.5"/>
      
      <rect x="270" y="360" width="200" height="40" rx="8" fill="url(#grd3)" filter="url(#sh)"/>
      <text x="370" y="384" textAnchor="middle" fill="#fff" fontSize="12" fontWeight="700">Fusion → Sigmoid → [0,1]</text>

      {/* Labels */}
      <text x="100" y="168" textAnchor="middle" fontSize="9" fill="#8b5cf6" fontWeight="600">GMF PATH</text>
      <text x="560" y="148" textAnchor="middle" fontSize="9" fill="#f59e0b" fontWeight="600">MLP PATH</text>
      <text x="370" y="414" textAnchor="middle" fontSize="11" fill="#64748b">User-Product Affinity Score</text>
    </svg>
  );
}

// ─── Styles ────────────────────────────────────────────────────────
const styles = {
  root: { display:"flex", minHeight:"100vh", background:"#f8fafc", fontFamily:"'DM Sans', 'Instrument Sans', system-ui, sans-serif" },
  sidebar: { width:220, background:"#0f172a", color:"#e2e8f0", display:"flex", flexDirection:"column", padding:"20px 0", position:"sticky", top:0, height:"100vh", flexShrink:0 },
  logo: { display:"flex", alignItems:"center", gap:10, padding:"0 20px", marginBottom:32 },
  logoIcon: { fontSize:28, color:"#818cf8" },
  logoTitle: { fontSize:16, fontWeight:700, color:"#f1f5f9", letterSpacing:-0.5 },
  logoSub: { fontSize:10, color:"#64748b", marginTop:1 },
  navGroup: { display:"flex", flexDirection:"column", gap:2, padding:"0 10px" },
  navBtn: { display:"flex", alignItems:"center", gap:10, padding:"10px 14px", background:"transparent", border:"none", color:"#94a3b8", fontSize:13, cursor:"pointer", borderRadius:8, textAlign:"left", transition:"all 0.2s" },
  navBtnActive: { background:"#1e293b", color:"#f1f5f9", fontWeight:600 },
  navIcon: { fontSize:16, width:20, textAlign:"center" },
  sidebarFooter: { marginTop:"auto", padding:"16px 20px", borderTop:"1px solid #1e293b" },
  footerBadge: { fontSize:11, color:"#818cf8", fontWeight:600 },

  main: { flex:1, overflow:"auto", minWidth:0 },
  page: { padding:"28px 32px", maxWidth:900, margin:"0 auto" },
  pageHeader: { marginBottom:28 },
  pageTitle: { fontSize:26, fontWeight:800, color:"#0f172a", letterSpacing:-0.8, margin:0, lineHeight:1.2 },
  pageSubtitle: { fontSize:14, color:"#64748b", marginTop:6 },

  statGrid: { display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:24 },
  statCard: { background:"#fff", borderRadius:14, padding:"20px 18px", boxShadow:"0 1px 3px rgba(0,0,0,0.06)", position:"relative", overflow:"hidden", animation:"fadeInUp 0.5s ease both" },
  statDot: { width:8, height:8, borderRadius:"50%", marginBottom:10 },
  statValue: { fontSize:28, fontWeight:800, color:"#0f172a", letterSpacing:-1 },
  statLabel: { fontSize:12, color:"#64748b", marginTop:2, fontWeight:500 },
  statDelta: { fontSize:11, color:"#94a3b8", marginTop:4 },

  twoCol: { display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:24 },
  card: { background:"#fff", borderRadius:14, padding:"22px 24px", boxShadow:"0 1px 3px rgba(0,0,0,0.06)" },
  cardTitle: { fontSize:15, fontWeight:700, color:"#1e293b", margin:"0 0 12px", letterSpacing:-0.3 },

  metricsGrid: { display:"flex", flexDirection:"column", gap:8, marginTop:8 },
  metricRow: { display:"flex", alignItems:"center", gap:10 },
  metricLabel: { fontSize:12, color:"#64748b", width:85, flexShrink:0 },
  metricBarOuter: { flex:1, height:6, background:"#f1f5f9", borderRadius:3, overflow:"hidden" },
  metricBarInner: { height:"100%", background:"linear-gradient(90deg,#6366f1,#a78bfa)", borderRadius:3, transition:"width 1.2s cubic-bezier(0.16, 1, 0.3, 1)" },
  metricValue: { fontSize:13, fontWeight:700, color:"#1e293b", width:40, textAlign:"right" },

  select: { width:"100%", padding:"10px 12px", borderRadius:8, border:"1px solid #e2e8f0", fontSize:13, color:"#1e293b", background:"#fff", outline:"none", cursor:"pointer" },
  profileGrid: { display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8 },
  filterLabel: { fontSize:11, fontWeight:600, color:"#64748b", textTransform:"uppercase", letterSpacing:0.5, display:"block", marginBottom:6 },
  filterChip: { padding:"6px 12px", borderRadius:20, border:"1px solid #e2e8f0", background:"#fff", fontSize:12, cursor:"pointer", display:"flex", alignItems:"center", transition:"all 0.2s" },
  filterChipActive: { background:"#6366f1", color:"#fff", borderColor:"#6366f1" },
  rangeInput: { width:"100%", accentColor:"#6366f1" },

  recGrid: { display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(240px, 1fr))", gap:16 },
  recCard: { background:"#fff", borderRadius:14, padding:"18px 20px", boxShadow:"0 1px 4px rgba(0,0,0,0.06)", border:"1px solid #f1f5f9", transition:"all 0.2s" },
  recHeader: { display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 },
  recBadge: { fontSize:10, fontWeight:600, padding:"3px 8px", borderRadius:20 },
  recCategory: { fontSize:11, color:"#6366f1", fontWeight:600, textTransform:"uppercase", letterSpacing:0.5 },
  recName: { fontSize:13, fontWeight:600, color:"#1e293b", marginTop:4, lineHeight:1.3 },
  recDetails: { display:"flex", gap:12, marginTop:12, paddingTop:10, borderTop:"1px solid #f1f5f9" },
  recDetail: { display:"flex", flexDirection:"column", gap:2 },
  recDetailLabel: { fontSize:10, color:"#94a3b8", textTransform:"uppercase" },
  recDetailValue: { fontSize:14, fontWeight:700, color:"#1e293b" },

  ctaButton: { display:"block", width:"100%", padding:"16px", background:"linear-gradient(135deg,#6366f1,#8b5cf6)", color:"#fff", border:"none", borderRadius:12, fontSize:15, fontWeight:700, cursor:"pointer", marginTop:24, transition:"transform 0.2s", textAlign:"center", letterSpacing:-0.3 },

  spinner: { width:32, height:32, border:"3px solid #e2e8f0", borderTopColor:"#6366f1", borderRadius:"50%", animation:"spin 0.8s linear infinite", margin:"0 auto" },

  codeBlock: { background:"#0f172a", color:"#e2e8f0", padding:"18px 20px", borderRadius:10, fontSize:12, lineHeight:1.6, overflow:"auto", fontFamily:"'JetBrains Mono', 'Fira Code', monospace", whiteSpace:"pre", margin:0 },
};

// Inject global keyframes
if (typeof document !== 'undefined') {
  const styleEl = document.createElement('style');
  styleEl.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeInUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
    * { box-sizing: border-box; }
    body { margin: 0; }
    button:hover { filter: brightness(1.05); }
    select:focus { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.1); }
  `;
  document.head.appendChild(styleEl);
}
