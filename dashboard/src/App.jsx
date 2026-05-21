import { useState, useEffect, useRef } from 'react';
import { Activity, Thermometer, CloudRain, Droplets, HeartPulse, Milk, Upload, History, ChevronDown, ChevronRight, Menu, AlertCircle, CheckCircle, AlertTriangle, Stethoscope, Zap, ShieldAlert, BrainCircuit, TrendingUp, TrendingDown, Sun, Waves, Scale, Cpu, BarChart3 } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, ComposedChart, AreaChart, Area, ScatterChart, Scatter, ReferenceLine, Cell,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, RadialBarChart, RadialBar
} from 'recharts';

function App() {
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const [activeTab, setActiveTab] = useState('farm');
  
  // Sidebar State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [openSections, setOpenSections] = useState({
    analytics: true,
    aiModels: true
  });

  // ---------------------------------------------
  // PHASE 13: LIVE ML INFERENCE STATE
  // ---------------------------------------------
  const [sensorTemp, setSensorTemp] = useState(42.5);
  const [sensorSwelling, setSensorSwelling] = useState(220);
  const [sensorResult, setSensorResult] = useState(null);
  const [isInferringSensor, setIsInferringSensor] = useState(false);
  const [visionResult, setVisionResult] = useState(null);
  const [isInferringVision, setIsInferringVision] = useState(false);
  const mastitisFileInputRef = useRef(null);

  const handleSensorInference = async () => {
    setIsInferringSensor(true);
    try {
      const response = await fetch(`${API_BASE}/api/infer_mastitis_sensor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ temperature: sensorTemp, swelling: sensorSwelling })
      });
      const result = await response.json();
      setSensorResult(result);
    } catch (e) {
      console.error(e);
    }
    setIsInferringSensor(false);
  };

  const handleVisionInference = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setIsInferringVision(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch(`${API_BASE}/api/infer_mastitis_vision`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      setVisionResult(result);
    } catch (e) {
      console.error(e);
    }
    setIsInferringVision(false);
  };

  // ---------------------------------------------  
  // Data States
  const [farmData, setFarmData] = useState(null);
  const [selectedFarmCow, setSelectedFarmCow] = useState(null);
  const [siresData, setSiresData] = useState(null);
  const [mastitisData, setMastitisData] = useState(null);
  const [epidemiologyData, setEpidemiologyData] = useState(null);
  
  // Tab-Specific States
  const [selectedGraph, setSelectedGraph] = useState('attrition');
  const [visionPrediction, setVisionPrediction] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const [mastitisPrediction, setMastitisPrediction] = useState(null);
  const [mastitisInputs, setMastitisInputs] = useState({ temperature: 38.5, hardness: 0, pain: 0, swelling: 200 });

  // ---------------------------------------------
  // PHASE 12: GENERAL DISEASE AI STATE
  // ---------------------------------------------
  const [diseaseInputs, setDiseaseInputs] = useState({
    temperature: 38.5,
    heart_rate: 70,
    respiratory_rate: 24,
    feed_quantity: 15.0,
    water_intake: 60.0,
    milk_yield: 25.0
  });
  const [diseaseResult, setDiseaseResult] = useState(null);
  const [isInferringDisease, setIsInferringDisease] = useState(false);

  // ---------------------------------------------
  // PHASE 13: MILK YIELD OPTIMIZER STATE
  // ---------------------------------------------
  const [yieldInputs, setYieldInputs] = useState({
    age_months: 48,
    weight_kg: 600,
    parity: 2,
    days_in_milk: 120,
    feed_quantity: 18.5,
    feeding_frequency: 3,
    water_intake: 80,
    prev_week_avg: 24.5,
    temp: 28,
    humidity: 65
  });
  const [yieldResult, setYieldResult] = useState(null);
  const [isInferringYield, setIsInferringYield] = useState(false);

  const handleYieldPredict = async () => {
    setIsInferringYield(true);
    setYieldResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/predict_yield`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(yieldInputs)
      });
      const result = await response.json();
      setYieldResult(result);
    } catch (e) {
      console.error(e);
    }
    setIsInferringYield(false);
  };

  const [isInferringSmart, setIsInferringSmart] = useState(false);
  const [smartResult, setSmartResult] = useState(null);
  const [smartInputs, setSmartInputs] = useState({
    body_temp: 38.5,
    heart_rate: 72,
    resp_rate: 28,
    ambient_temp: 28,
    humidity: 65,
    feed_qty: 15,
    feeding_frequency: 3,
    water_intake: 60,
    days_in_milk: 120,
    prev_week_avg: 22,
    actual_yield: 21.5
  });

  const handleSmartInference = async () => {
    setIsInferringSmart(true);
    try {
      const response = await fetch(`${API_BASE}/api/ai_smart_inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(smartInputs)
      });
      const data = await response.json();
      setSmartResult(data);
    } catch (err) {
      console.error("Inference Error:", err);
    } finally {
      setIsInferringSmart(false);
    }
  };

  // ---------------------------------------------
  // NEW: Disease-Conditioned Yield Impact State
  // ---------------------------------------------
  const [diseaseOptions, setDiseaseOptions] = useState(["Healthy", "Mastitis", "Ketosis", "Foot_and_Mouth_Disease"]);
  const [impactDisease, setImpactDisease] = useState("Mastitis");
  const [impactResult, setImpactResult] = useState(null);
  const [isInferringImpact, setIsInferringImpact] = useState(false);

  const handleImpactInference = async () => {
    setIsInferringImpact(true);
    setImpactResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/predict_disease_impact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          body_temp: smartInputs.body_temp,
          heart_rate: smartInputs.heart_rate,
          resp_rate: smartInputs.resp_rate,
          ambient_temp: smartInputs.ambient_temp,
          humidity: smartInputs.humidity,
          feed_qty: smartInputs.feed_qty,
          target_disease: impactDisease
        })
      });
      const data = await response.json();
      setImpactResult(data);
    } catch (err) {
      console.error("Impact Inference Error:", err);
    } finally {
      setIsInferringImpact(false);
    }
  };

  // ---------------------------------------------
  // NEW: Multi-Class Symptom Checker State
  // ----------------------------------------
  const [isInferringSymptoms, setIsInferringSymptoms] = useState(false);
  const [symptomResult, setSymptomResult] = useState(null);
  const [symptomInputs, setSymptomInputs] = useState({
    body_temp: 39.8,
    heart_rate: 90,
    resp_rate: 45,
    feed_qty: 8.0,
    milk_yield: 5.5,
    ambient_temp: 35.0,
    humidity: 80.0
  });

  const handleSymptomInference = async () => {
    setIsInferringSymptoms(true);
    setSymptomResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/diagnose_symptoms`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(symptomInputs)
      });
      if (!response.ok) {
        throw new Error(`API returned ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      if (data.top_predictions) {
        setSymptomResult(data);
      } else {
        throw new Error("Invalid response format from API");
      }
    } catch (err) {
      console.error("Symptom Inference Error:", err);
      setSymptomResult({ error: err.message });
    } finally {
      setIsInferringSymptoms(false);
    }
  };

  // ---------------------------------------------
  // NEW: Future Lactation Predictor State
  // ---------------------------------------------
  const [isInferringPeak, setIsInferringPeak] = useState(false);
  const [peakResult, setPeakResult] = useState(null);
  const [peakInputs, setPeakInputs] = useState({
    lactation_number: 2,
    length_of_lactation: 265,
    days_dry: 60,
    total_milk_yield: 6500.5
  });

  const handlePeakPredict = async () => {
    setIsInferringPeak(true);
    setPeakResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/predict_next_peak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(peakInputs)
      });
      if (!response.ok) throw new Error("API failed");
      const data = await response.json();
      setPeakResult(data);
    } catch (err) {
      console.error("Peak Predictor Error:", err);
    } finally {
      setIsInferringPeak(false);
    }
  };

  // ---------------------------------------------
  // NEW: Milk Quality Predictor State
  // ---------------------------------------------
  const [isInferringQuality, setIsInferringQuality] = useState(false);
  const [qualityResult, setQualityResult] = useState(null);
  const [qualityInputs, setQualityInputs] = useState({
    breed: 'Holstein',
    parity: 2,
    calving_season: 'Spring',
    calv_int: 380
  });

  const handleQualityPredict = async () => {
    setIsInferringQuality(true);
    setQualityResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/predict_milk_quality`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(qualityInputs)
      });
      if (!response.ok) throw new Error("API failed");
      const data = await response.json();
      setQualityResult(data);
    } catch (err) {
      console.error("Quality Predictor Error:", err);
    } finally {
      setIsInferringQuality(false);
    }
  };

  useEffect(() => {
    fetch(`${API_BASE}/api/farm_live`)
      .then(res => res.json())
      .then(data => {
        setFarmData(data);
        if (data.cows && data.cows.length > 0) setSelectedFarmCow(data.cows[0]);
      })
      .catch(err => console.error("Error fetching live farm API:", err));

    fetch(`${API_BASE}/api/sires_analytics`)
      .then(res => res.json())
      .then(data => setSiresData(data))
      .catch(err => console.error("Error fetching sires analytics API:", err));

    fetch(`${API_BASE}/api/mastitis_analytics`)
      .then(res => res.json())
      .then(data => setMastitisData(data))
      .catch(err => console.error("Error fetching mastitis analytics API:", err));

    fetch(`${API_BASE}/api/global_disease_analytics`)
      .then(res => res.json())
      .then(data => setEpidemiologyData(data))
      .catch(err => console.error("Error fetching epidemiology API:", err));

    fetch(`${API_BASE}/api/diseases`)
      .then(res => res.json())
      .then(data => setDiseaseOptions(data))
      .catch(err => console.error("Error fetching diseases list:", err));
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    fetch(`${API_BASE}/api/predict_vision`, { method: 'POST', body: formData })
      .then(res => res.json())
      .then(data => { setVisionPrediction(data.predicted_yield_kg); setIsUploading(false); })
      .catch(err => { console.error(err); setIsUploading(false); });
  };

  const handleMastitisPredict = () => {
    fetch(`${API_BASE}/api/predict_mastitis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(mastitisInputs)
    })
    .then(res => res.json())
    .then(data => setMastitisPrediction(data.mastitis_risk))
    .catch(err => console.error("Error predicting mastitis:", err));
  };

  if (!farmData || !siresData) return <div style={{ color: 'white', padding: '2rem', display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', fontSize: '1.5rem' }}>Synchronizing Dairy Datasets...</div>;

  // ------------ TAB 1: SIMPLE FARMER DASHBOARD ------------
  const renderFarmTab = () => {
    return (
      <div className="farm-tab" style={{ padding: '0 1rem', animation: 'fade-in 0.5s' }}>
        
        {/* Alerts Banner */}
        {farmData.alerts.length > 0 && (
          <div style={{ backgroundColor: 'rgba(207, 102, 121, 0.15)', borderLeft: '4px solid var(--danger)', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
            <h4 style={{ margin: '0 0 10px 0', color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '8px' }}><AlertCircle size={20}/> Daily Anomalies & Alerts</h4>
            {farmData.alerts.map((alert, i) => <div key={i} style={{ color: 'white', fontSize: '0.9rem', marginBottom: '4px' }}>• {alert}</div>)}
          </div>
        )}

        {/* Traffic Light Health Overview */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
           <div className="glass-panel" style={{ textAlign: 'center', borderBottom: '6px solid #4CAF50', padding: '1.5rem' }}>
              <CheckCircle color="#4CAF50" size={36} style={{marginBottom: '10px'}}/>
              <h2 style={{margin:0, color: 'white', fontSize: '2.5rem'}}>{farmData.counts.green}</h2>
              <p style={{color: 'var(--text-main)', margin: '5px 0 0 0', fontWeight: 'bold'}}>Healthy Cows</p>
           </div>
           <div className="glass-panel" style={{ textAlign: 'center', borderBottom: '6px solid #ffeb3b', padding: '1.5rem' }}>
              <AlertTriangle color="#ffeb3b" size={36} style={{marginBottom: '10px'}}/>
              <h2 style={{margin:0, color: 'white', fontSize: '2.5rem'}}>{farmData.counts.yellow}</h2>
              <p style={{color: 'var(--text-main)', margin: '5px 0 0 0', fontWeight: 'bold'}}>Medium Risk (Monitor)</p>
           </div>
           <div className="glass-panel" style={{ textAlign: 'center', borderBottom: '6px solid var(--danger)', padding: '1.5rem' }}>
              <AlertCircle color="var(--danger)" size={36} style={{marginBottom: '10px'}}/>
              <h2 style={{margin:0, color: 'white', fontSize: '2.5rem'}}>{farmData.counts.red}</h2>
              <p style={{color: 'var(--text-main)', margin: '5px 0 0 0', fontWeight: 'bold'}}>High Risk (Action Needed)</p>
           </div>
        </div>

        {/* SECTION A: HERD LEVEL INSIGHTS */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem', marginBottom: '2rem' }}>
          
          {/* Chart 3: Fever vs Milk Drop (BarChart) */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 style={{color: 'white', margin: '0 0 1rem 0'}}>3. Fever vs Milk Drop</h3>
            <p style={{color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '1rem'}}>Herd-wide average comparison.</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={farmData.herd_stats.fever_comparison} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="fever" stroke="var(--text-main)" label={{ value: 'Health Status', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40} />
                <YAxis stroke="var(--text-main)" domain={['dataMin - 5', 'dataMax + 5']} label={{ value: 'Avg Milk (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--danger)' }} />
                <Bar dataKey="avg_milk" fill="var(--danger)" radius={[4, 4, 0, 0]}>
                   {farmData.herd_stats.fever_comparison.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={entry.fever === 'Healthy' ? '#4CAF50' : 'var(--danger)'} />
                   ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 7: Cow Performance Comparison (BarChart) */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 style={{color: 'white', margin: '0 0 1rem 0'}}>7. Cow Performance Comparison</h3>
            <p style={{color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '1rem'}}>Top to bottom performers based on average daily yield.</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={farmData.herd_stats.cow_performance} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="cow_id" stroke="var(--text-main)" label={{ value: 'Cow ID Number', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40} />
                <YAxis stroke="var(--text-main)" domain={['dataMin - 2', 'dataMax + 2']} label={{ value: 'Avg Milk (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)' }} />
                <Bar dataKey="avg_milk" fill="var(--text-highlight)" radius={[4, 4, 0, 0]} name="Average Milk" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Cow Cards Database */}
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{color: 'var(--text-highlight)', margin: '0 0 15px 0'}}>Live Tracker Roster (Select Cow)</h3>
          <div style={{ display: 'flex', gap: '15px', overflowX: 'auto', paddingBottom: '15px', scrollbarColor: 'var(--accent-color) transparent' }}>
            {farmData.cows.map(cow => (
               <div 
                 key={cow.cow_id} 
                 onClick={() => setSelectedFarmCow(cow)}
                 style={{ 
                   minWidth: '220px', 
                   backgroundColor: selectedFarmCow?.cow_id === cow.cow_id ? 'rgba(100, 255, 218, 0.1)' : 'rgba(11,12,16,0.6)',
                   border: `2px solid ${cow.status === 'red' ? 'var(--danger)' : cow.status === 'yellow' ? '#ffeb3b' : '#4CAF50'}`,
                   padding: '1.2rem', borderRadius: '12px', cursor: 'pointer', transition: '0.2s',
                   boxShadow: selectedFarmCow?.cow_id === cow.cow_id ? '0 0 15px rgba(100,255,218,0.2)' : 'none'
                 }}>
                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                    <strong style={{color: 'white', fontSize: '1.3rem'}}>{cow.cow_id} 🐄</strong>
                    <div style={{ width: '18px', height: '18px', borderRadius: '50%', backgroundColor: cow.status === 'red' ? 'var(--danger)' : cow.status === 'yellow' ? '#ffeb3b' : '#4CAF50', boxShadow: '0 0 8px rgba(0,0,0,0.5)' }}></div>
                  </div>
                  <div style={{color: 'var(--text-main)', marginTop: '12px', fontSize: '0.95rem'}}>Today's Milk: <span style={{color: 'white', fontWeight: 'bold'}}>{cow.today_milk} kg</span></div>
                  <div style={{color: cow.status === 'red' ? 'var(--danger)' : cow.status === 'yellow' ? '#ffeb3b' : '#4CAF50', fontSize: '0.9rem', marginTop: '8px', fontWeight: 700}}>
                     {cow.status_msg}
                  </div>
               </div>
            ))}
          </div>
        </div>

        {/* Graphing Viewport for Selected Cow */}
        {selectedFarmCow && (
          <div className="glass-panel" style={{ padding: '2rem' }}>
             <h2 style={{color: 'white', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem', marginBottom: '2rem' }}>
                Individual Analysis: {selectedFarmCow.cow_id}
             </h2>
             
             {/* ROW 1: Lactation Curve & Health Timeline */}
             <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', marginBottom: '3rem' }}>
                
                {/* Chart 1: Lactation Curve (Milk vs DIM) */}
                <div>
                  <h4 style={{color: '#fff', marginBottom: '1rem'}}>1. Lactation Curve (Milk vs DIM)</h4>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={selectedFarmCow.history} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="day" stroke="var(--text-main)" label={{ value: 'Timeline (Days In Milk)', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40}/>
                      <YAxis stroke="var(--text-main)" domain={['dataMin - 2', 'dataMax + 2']} label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                      <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: '#4CAF50', color: 'white' }} />
                      <Line type="monotone" dataKey="actual_milk" stroke="#4CAF50" strokeWidth={3} name="Milk Yield (kg)" dot={{r: 4}} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Chart 5: Health Timeline (Per Cow) */}
                <div>
                  <h4 style={{color: '#fff', marginBottom: '1rem'}}>5. Health Timeline (Events)</h4>
                  <ResponsiveContainer width="100%" height={260}>
                    <ComposedChart data={selectedFarmCow.history} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="timestamp" stroke="var(--text-main)" tick={false} label={{ value: 'Date / Time', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40}/>
                      <YAxis stroke="var(--text-main)" domain={['dataMin - 2', 'dataMax + 2']} label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                      <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--danger)', color: 'white' }} />
                      <Legend verticalAlign="top" height={30} />
                      <Line type="monotone" dataKey="actual_milk" stroke="rgba(255,255,255,0.2)" strokeWidth={2} name="Milk Line" dot={false} />
                      <Scatter dataKey="fever" fill="var(--danger)" name="Fever Occurred" />
                      <Scatter dataKey="lameness" fill="#ffeb3b" name="Lameness/Lethargy" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
             </div>

             {/* ROW 2: Heat Stress & Activity Scatter */}
             <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', marginBottom: '3rem' }}>
                
                {/* Chart 2: Heat Stress Impact */}
                <div>
                  <h4 style={{color: '#fff', marginBottom: '1rem'}}>2. Heat Stress Impact</h4>
                  <ResponsiveContainer width="100%" height={260}>
                    <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="thi" type="number" name="THI" domain={['dataMin - 2', 'dataMax + 2']} stroke="var(--text-main)" label={{ value: 'THI Heat Index (Units)', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40} />
                      <YAxis dataKey="actual_milk" type="number" name="Milk (kg)" stroke="var(--text-main)" label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                      <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--danger)', color: 'white' }} />
                      <ReferenceLine x={72} stroke="var(--danger)" strokeDasharray="3 3" label={{ position: 'top', value: 'Danger Threshold > 72', fill: 'var(--danger)' }} />
                      <Scatter name="Yield vs THI" data={selectedFarmCow.history} fill="var(--text-highlight)">
                        {selectedFarmCow.history.map((entry, index) => (
                           <Cell key={`cell-${index}`} fill={entry.heat_stress === 1 ? 'var(--danger)' : 'var(--text-highlight)'} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* Chart 4: Activity vs Milk Relationship */}
                <div>
                  <h4 style={{color: '#fff', marginBottom: '1rem'}}>4. Activity vs Milk Relationship</h4>
                  <ResponsiveContainer width="100%" height={260}>
                    <ScatterChart margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="activity" type="number" name="Activity" domain={['dataMin - 5', 'dataMax + 5']} stroke="var(--text-main)" label={{ value: '3D Activity Index (Movement Units)', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40} />
                      <YAxis dataKey="actual_milk" type="number" name="Milk (kg)" stroke="var(--text-main)" label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                      <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: '#ffeb3b', color: 'white' }} />
                      <Scatter name="Activity vs Yield" data={selectedFarmCow.history} fill="#ffeb3b" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
             </div>

             {/* ROW 3: Chart 6 - Environmental Dual-Axis */}
             <div>
                <h4 style={{color: '#fff', marginBottom: '1rem'}}>6. Environmental Impact Dashboard (Dual-Axis)</h4>
                <ResponsiveContainer width="100%" height={320}>
                  <ComposedChart data={selectedFarmCow.history} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="timestamp" stroke="var(--text-main)" tick={false} label={{ value: 'Timeline (Chronological Dates)', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40} />
                    <YAxis yAxisId="left" stroke="#4CAF50" label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: '#4CAF50' }} width={60} />
                    <YAxis yAxisId="right" orientation="right" stroke="var(--danger)" domain={[60, 85]} label={{ value: 'THI Index (Units)', angle: 90, position: 'insideRight', offset: 15, fill: 'var(--danger)' }} width={60} />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: 'white' }} />
                    <Legend verticalAlign="top" height={36} />
                    <Line yAxisId="left" type="monotone" dataKey="actual_milk" stroke="#4CAF50" strokeWidth={3} name="Actual Milk Yield (kg)" dot={false} />
                    <Line yAxisId="right" type="monotone" dataKey="thi" stroke="var(--danger)" strokeWidth={2} strokeDasharray="5 5" name="Barn THI Index" dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
             </div>

             {/* ROW 4: Chart 8 - Actual vs Expected Yield */}
             <div style={{ marginTop: '3rem' }}>
                <h4 style={{color: '#fff', marginBottom: '1rem'}}>8. Target Deficits (Actual vs Expected Yield & Fever Drops)</h4>
                <p style={{color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '1rem'}}>Compare actual milk produced vs mathematically expected target. Red dots indicate Fever Days dragging the average down.</p>
                <ResponsiveContainer width="100%" height={320}>
                  <ComposedChart data={selectedFarmCow.history} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="day" stroke="var(--text-main)" label={{ value: 'Timeline (Days In Milk)', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={50} />
                    <YAxis stroke="var(--text-main)" domain={['dataMin - 5', 'dataMax + 5']} label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--text-main)' }} width={60} />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: 'white' }} />
                    <Legend verticalAlign="top" height={36} />
                    <Line type="monotone" dataKey="actual_milk" stroke="#4CAF50" strokeWidth={4} name="Actual Milk Produced (kg)" dot={false} />
                    <Line type="monotone" dataKey="expected_milk" stroke="rgba(255,255,255,0.4)" strokeWidth={2} strokeDasharray="5 5" name="Expected Target Average (kg)" dot={false} />
                    <Scatter dataKey="fever" fill="var(--danger)" name="Fever Occurred (Yield Dropped)" shape="circle" />
                  </ComposedChart>
                </ResponsiveContainer>
             </div>

          </div>
        )}
      </div>
    );
  };

  // ------------ TAB 2: RESEARCH / ANALYTICS ------------
  const renderResearchTab = () => {
    const renderResearchGraph = () => {
      switch (selectedGraph) {
        case 'attrition':
          return (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={siresData.attrition_funnel} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-main)" label={{ value: 'Lactation Cycle', position: 'insideBottom', offset: -5, fill: 'var(--text-main)' }} height={50} />
                <YAxis stroke="var(--text-main)" label={{ value: 'Number of Cows', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)', color: '#fff' }} />
                <Legend />
                <Bar dataKey="count" name="Surviving Herd Count" fill="var(--danger)" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          );
        case 'dry_period':
          return (
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 30, left: 40, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="Days_Dry" type="number" stroke="var(--text-main)" name="Days Dry" label={{ value: 'Days Dry Before Calving', position: 'insideBottom', offset: -10, fill: 'var(--text-main)' }} />
                <YAxis dataKey="Yield" type="number" stroke="var(--text-main)" name="Milk Yield (kg)" label={{ value: 'Total Milk Yield (kg)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-highlight)', color: '#fff' }} />
                <Scatter name="Cows" data={siresData.dry_period} fill="var(--text-highlight)" />
              </ScatterChart>
            </ResponsiveContainer>
          );
        case 'persistency':
          return (
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 20, right: 30, left: 40, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="Peak" type="number" stroke="var(--text-main)" name="Peak Yield (kg)" label={{ value: 'Peak Milk Yield (kg)', position: 'insideBottom', offset: -10, fill: 'var(--text-main)' }} />
                <YAxis dataKey="Total" type="number" stroke="var(--text-main)" name="Total Yield (kg)" label={{ value: 'Total Milk Yield (kg)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)', color: '#fff' }} />
                <Scatter name="Cows" data={siresData.persistency} fill="var(--accent-color)" />
              </ScatterChart>
            </ResponsiveContainer>
          );
        case 'lifecycle':
          return (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={siresData.lifecycle} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="name" stroke="var(--text-main)" label={{ value: 'Lactation Cycle', position: 'insideBottom', offset: -5, fill: 'var(--text-main)' }} height={50} />
                <YAxis yAxisId="left" stroke="var(--text-highlight)" label={{ value: 'Peak Yield (kg)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                <YAxis yAxisId="right" orientation="right" stroke="var(--danger)" label={{ value: 'Days', angle: 90, position: 'insideRight', fill: 'var(--text-main)' }} />
                <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                <Legend iconType="circle" />
                <Line yAxisId="left" type="monotone" dataKey="Peak_Yield" stroke="var(--text-highlight)" strokeWidth={4} dot={{ r: 6 }} name="Peak Yield (kg)" />
                <Line yAxisId="right" type="monotone" dataKey="Days_to_Peak" stroke="var(--danger)" strokeWidth={3} dot={{ r: 5 }} name="Days To Peak" />
                <Line yAxisId="right" type="monotone" dataKey="Dry_Days" stroke="var(--text-main)" strokeWidth={2} strokeDasharray="5 5" name="Dry Rest Days" />
              </LineChart>
            </ResponsiveContainer>
          );
        case 'cohorts':
          const surv = siresData.cohorts.survival;
          return (
            <div style={{width: '100%'}}>
               <div style={{ display: 'flex', justifyContent: 'space-between', padding: '15px 20px', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '8px', marginBottom: '20px' }}>
                  <div style={{textAlign: 'center'}}><span style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Started L1</span><br/><strong style={{color: 'white', fontSize: '1.2rem'}}>{surv.L1_Count} Cows</strong></div>
                  <div style={{textAlign: 'center'}}><span style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Survive to L2</span><br/><strong style={{color: 'var(--accent-color)', fontSize: '1.2rem'}}>{surv.L2_Count} ({surv.L2_Pct}%)</strong></div>
                  <div style={{textAlign: 'center'}}><span style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Survive to L3</span><br/><strong style={{color: 'var(--text-highlight)', fontSize: '1.2rem'}}>{surv.L3_Count} ({surv.L3_Pct}%)</strong></div>
                  <div style={{textAlign: 'center'}}><span style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Elite L4</span><br/><strong style={{color: 'var(--danger)', fontSize: '1.2rem'}}>{surv.L4_Count} ({surv.L4_Pct}%)</strong></div>
                  <div style={{textAlign: 'center'}}><span style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Total Analyzed</span><br/><strong style={{color: 'white', fontSize: '1.2rem'}}>{surv.Total_Unique} Herd</strong></div>
               </div>
               <ResponsiveContainer width="100%" height={380}>
                 <ComposedChart data={siresData.cohorts.graph} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                   <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                   <XAxis dataKey="name" stroke="var(--text-main)" label={{ value: 'Lactation Cycle', position: 'insideBottom', offset: -5, fill: 'var(--text-main)' }} height={50} />
                   <YAxis stroke="var(--text-main)" domain={['dataMin - 1000', 'dataMax + 1000']} label={{ value: 'Milk Yield (kg)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                   <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)', color: '#fff' }} />
                   <Legend />
                   <Area type="monotone" dataKey="L5_Bounds" fill="rgba(207, 102, 121, 0.2)" stroke="none" name="AI 95% Confidence Band" />
                   <Line type="monotone" dataKey="Cohort_2" stroke="var(--text-main)" strokeWidth={2} dot={{r: 4}} name="2-Lactation Cows" />
                   <Line type="monotone" dataKey="Cohort_3" stroke="var(--accent-color)" strokeWidth={3} dot={{r: 5}} name="3-Lactation Cows" />
                   <Line type="monotone" dataKey="Elite_75" stroke="var(--text-highlight)" strokeWidth={4} dot={{r: 6}} name="Elite 75 Herd (L1-L4)" />
                   <Line type="monotone" dataKey="L5_Prediction" stroke="var(--danger)" strokeWidth={4} strokeDasharray="5 5" dot={{r: 8, fill: 'var(--danger)'}} name="AI L5 Projection" />
                 </ComposedChart>
               </ResponsiveContainer>
            </div>
          );
        default:
          return <div style={{color:'white', textAlign:'center', marginTop:'50px'}}>Select a tool.</div>;
      }
    };

    return (
      <div className="glass-panel" style={{ margin: '1rem', animation: 'fade-in 0.5s' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
           <div>
             <h2 style={{ margin: 0, color: 'var(--text-highlight)' }}>Deep Research Hub</h2>
             <p style={{ color: 'var(--text-main)', margin: '5px 0 0 0' }}>Data science and genetic regression visualizations extracted from `Superior Sires Excel`.</p>
           </div>
           <div style={{ position: 'relative' }}>
              <select 
                style={{
                  appearance: 'none', backgroundColor: 'rgba(255,255,255,0.05)', color: 'white',
                  border: '1px solid var(--text-main)', padding: '12px 40px 12px 20px',
                  borderRadius: '8px', fontSize: '1rem', cursor: 'pointer', outline: 'none'
                }}
                value={selectedGraph}
                onChange={(e) => setSelectedGraph(e.target.value)}
              >
                <option value="attrition" style={{color: "black"}}>1. Cow Survival Rate Over Time</option>
                <option value="dry_period" style={{color: "black"}}>2. Dry Period Optimization</option>
                <option value="persistency" style={{color: "black"}}>3. Lactation Persistency Curves</option>
                <option value="lifecycle" style={{color: "black"}}>4. Lifecycle Efficiency Track</option>
                <option value="cohorts" style={{color: "black"}}>5. AI Predictive Cohorts (L1-L5)</option>
              </select>
              <ChevronDown size={18} color="white" style={{ position: 'absolute', right: '15px', top: '14px', pointerEvents: 'none' }} />
           </div>
        </div>
        <div className="graph-viewer" style={{ width: '100%', minHeight: 450 }}>
          {renderResearchGraph()}
        </div>
      </div>
    );
  };

  // ------------ TAB 3: VISION ------------
  const renderVisionTab = () => {
    return (
      <div className="glass-panel" style={{ margin: '3rem auto', maxWidth: '800px', textAlign: 'center', animation: 'fade-in 0.5s' }}>
        <h2 style={{color: 'white', marginBottom: '10px'}}>Visual Yield Predictor</h2>
        <p style={{color: 'var(--text-main)', marginBottom: '3rem'}}>Upload a rear udder photo of your cow to let the AI visually estimate her milk yield capacity. <br/><strong style={{color: 'var(--danger)'}}>Warning: Please use ONLY rear udder images. Full body images will cause inaccurate yield predictions.</strong></p>
        
        <div style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px', padding: '4rem', border: '2px dashed var(--text-highlight)', borderRadius: '16px', background: 'rgba(100, 255, 218, 0.05)' }} onClick={() => fileInputRef.current.click()}>
          <Upload size={64} color="var(--text-highlight)" />
          <h3 style={{color: 'white', margin: 0}}>{isUploading ? "Analyzing Image..." : "Drag & Drop Image Here"}</h3>
          <span style={{color: 'var(--text-main)'}}>Supports .jpg, .png, .jpeg</span>
        </div>
        <input type="file" accept="image/*" ref={fileInputRef} style={{ display: 'none' }} onChange={handleImageUpload} />
        
        {visionPrediction && !isUploading && (
          <div style={{ marginTop: '3rem', padding: '2rem', backgroundColor: 'rgba(207, 102, 121, 0.1)', borderRadius: '12px', border: '1px solid var(--danger)', animation: 'fade-in 1s' }}>
            <h4 style={{ color: 'var(--text-main)', margin: '0 0 10px 0', textTransform: 'uppercase', letterSpacing: '2px' }}>AI Estimated Yield</h4>
            <div style={{ fontSize: '4rem', fontWeight: 900, color: 'var(--danger)', textShadow: '0 0 20px rgba(207,102,121,0.6)' }}>{visionPrediction} kg</div>
          </div>
        )}
      </div>
    );
  };

  // ------------ TAB 4: MASTITIS ------------
  const renderMastitisTab = () => {
    if (!mastitisData) return <div style={{color:'white', padding:'2rem'}}>Loading Mastitis Models...</div>;
    return (
      <div className="glass-panel" style={{ margin: '1rem', animation: 'fade-in 0.5s', padding: '2rem' }}>
        <h2 style={{color: 'var(--danger)', marginBottom: '10px'}}>🩺 Mastitis Early Warning</h2>
        <p style={{color: 'var(--text-main)', marginBottom: '3rem'}}>Using smart barn sensors to detect sick cows before they show symptoms.</p>
        
        {/* NEW PHASE 12: REAL-TIME DIAGNOSIS ENGINE */}
        <div className="glass-panel" style={{ padding: '2rem', marginBottom: '3rem', border: mastitisPrediction !== null ? (mastitisPrediction > 0.5 ? '2px solid var(--danger)' : '2px solid #4CAF50') : '1px solid rgba(255,255,255,0.1)' }}>
           <h3 style={{color: 'white', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px'}}><Activity color="var(--text-highlight)"/> Live Cow Diagnosis</h3>
           
           <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', marginBottom: '2rem' }}>
              <div>
                 <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '8px'}}>Core Temperature (°C)</label>
                 <input type="number" min="20" max="60" step="0.1" value={mastitisInputs.temperature} onChange={(e) => setMastitisInputs({...mastitisInputs, temperature: parseFloat(e.target.value) || 0})} style={{width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--text-main)', backgroundColor: 'rgba(255,255,255,0.05)', color: 'white', fontSize: '1.1rem'}} />
              </div>
              <div>
                 <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '8px'}}>Avg Flex Swelling (ADC)</label>
                 <input type="number" min="150" max="400" value={mastitisInputs.swelling} onChange={(e) => setMastitisInputs({...mastitisInputs, swelling: parseFloat(e.target.value) || 0})} style={{width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--text-main)', backgroundColor: 'rgba(255,255,255,0.05)', color: 'white', fontSize: '1.1rem'}} />
              </div>
              <div>
                 <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '8px'}}>Severe Udder Hardness?</label>
                 <select value={mastitisInputs.hardness} onChange={(e) => setMastitisInputs({...mastitisInputs, hardness: parseInt(e.target.value)})} style={{width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--text-main)', backgroundColor: 'rgba(255,255,255,0.05)', color: 'white', fontSize: '1.1rem'}}>
                    <option value={0} style={{color: 'black'}}>No (Normal)</option>
                    <option value={1} style={{color: 'black'}}>Yes (Hard/Abnormal)</option>
                 </select>
              </div>
              <div>
                 <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '8px'}}>Signs of Severe Pain?</label>
                 <select value={mastitisInputs.pain} onChange={(e) => setMastitisInputs({...mastitisInputs, pain: parseInt(e.target.value)})} style={{width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--text-main)', backgroundColor: 'rgba(255,255,255,0.05)', color: 'white', fontSize: '1.1rem'}}>
                    <option value={0} style={{color: 'black'}}>No (Normal)</option>
                    <option value={1} style={{color: 'black'}}>Yes (Pain Response)</option>
                 </select>
              </div>
           </div>
           
           <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
             <button onClick={handleMastitisPredict} style={{ padding: '14px 30px', backgroundColor: 'var(--text-highlight)', color: 'black', fontWeight: 'bold', fontSize: '1.1rem', border: 'none', borderRadius: '8px', cursor: 'pointer', transition: '0.2s' }}>
                Run XGBoost Diagnostic
             </button>
             
             {mastitisPrediction !== null && (
               <div style={{ display: 'flex', alignItems: 'center', gap: '15px', animation: 'fade-in 0.5s' }}>
                 <div style={{ fontSize: '2rem', fontWeight: 900, color: mastitisPrediction > 0.5 ? 'var(--danger)' : '#4CAF50' }}>
                   {(mastitisPrediction * 100).toFixed(1)}% Risk
                 </div>
                 <div style={{ color: 'var(--text-main)', fontSize: '1.1rem' }}>
                   {mastitisPrediction > 0.5 ? '🚨 High Probability of Clinical Mastitis' : '✅ Healthy Baseline Detected'}
                 </div>
               </div>
             )}
           </div>
        </div>

        {/* ------------------------------------------------ */}
        {/* PHASE 13: LIVE AI INFERENCE STATION */}
        {/* ------------------------------------------------ */}
        <div style={{ marginBottom: '4rem', padding: '2rem', background: 'rgba(100, 255, 218, 0.03)', borderRadius: '16px', border: '1px solid var(--text-highlight)' }}>
           <h3 style={{color: 'var(--text-highlight)', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px'}}><Activity size={24}/> Live ML Inference Station</h3>
           
           <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem' }}>
              {/* Sensor Panel */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h4 style={{ color: 'white', marginBottom: '1rem' }}>📡 Tabular Sensor Input (XGBoost)</h4>
                <div style={{ marginBottom: '1.5rem' }}>
                   <label style={{ color: 'var(--text-main)', display: 'block', marginBottom: '10px' }}>Cow Core Temperature: <strong style={{color: sensorTemp > 44 ? 'var(--danger)' : 'white'}}>{sensorTemp}°C</strong></label>
                   <input type="range" min="35" max="50" step="0.1" value={sensorTemp} onChange={(e) => setSensorTemp(parseFloat(e.target.value))} style={{ width: '100%', accentColor: 'var(--text-highlight)' }} />
                </div>
                <div style={{ marginBottom: '2rem' }}>
                   <label style={{ color: 'var(--text-main)', display: 'block', marginBottom: '10px' }}>Udder Swelling (ADC Flex Limit): <strong style={{color: sensorSwelling > 250 ? 'var(--danger)' : 'white'}}>{sensorSwelling} Units</strong></label>
                   <input type="range" min="150" max="300" step="1" value={sensorSwelling} onChange={(e) => setSensorSwelling(parseInt(e.target.value))} style={{ width: '100%', accentColor: 'var(--text-highlight)' }} />
                </div>
                <button onClick={handleSensorInference} disabled={isInferringSensor} style={{ width: '100%', padding: '12px', background: 'var(--text-highlight)', color: '#000', border: 'none', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer' }}>
                   {isInferringSensor ? "Processing..." : "Run XGBoost Inference"}
                </button>
                
                {sensorResult && (
                  <div style={{ marginTop: '1.5rem', padding: '1rem', background: sensorResult.probability > 50 ? 'rgba(207, 102, 121, 0.2)' : 'rgba(76, 175, 80, 0.2)', borderRadius: '8px', border: `1px solid ${sensorResult.probability > 50 ? 'var(--danger)' : '#4CAF50'}` }}>
                     <h5 style={{ margin: 0, color: 'white' }}>Diagnostic: {sensorResult.diagnosis}</h5>
                     <p style={{ margin: '5px 0 0 0', color: sensorResult.probability > 50 ? 'var(--danger)' : '#4CAF50', fontWeight: 'bold' }}>Probability: {sensorResult.probability}%</p>
                  </div>
                )}
              </div>

              {/* Vision Panel */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h4 style={{ color: 'white', marginBottom: '1rem' }}>📷 Milk Vision Upload (CNN)</h4>
                <div style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px', padding: '2rem', border: '2px dashed var(--text-main)', borderRadius: '12px', background: 'rgba(255, 255, 255, 0.02)' }} onClick={() => mastitisFileInputRef.current.click()}>
                  <Upload size={32} color="var(--text-main)" />
                  <span style={{color: 'white', fontWeight: 'bold'}}>{isInferringVision ? "Running CNN Pass..." : "Upload Milk Image"}</span>
                  <span style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>.keras Model Inference</span>
                </div>
                <input type="file" accept="image/*" ref={mastitisFileInputRef} style={{ display: 'none' }} onChange={handleVisionInference} />
                
                {visionResult && !isInferringVision && (
                  <div style={{ marginTop: '1.5rem', padding: '1rem', background: visionResult.probability > 50 ? 'rgba(207, 102, 121, 0.2)' : 'rgba(76, 175, 80, 0.2)', borderRadius: '8px', border: `1px solid ${visionResult.probability > 50 ? 'var(--danger)' : '#4CAF50'}` }}>
                     <h5 style={{ margin: 0, color: 'white' }}>Diagnostic: {visionResult.diagnosis}</h5>
                     <p style={{ margin: '5px 0 0 0', color: visionResult.probability > 50 ? 'var(--danger)' : '#4CAF50', fontWeight: 'bold' }}>Probability: {visionResult.probability}%</p>
                  </div>
                )}
              </div>
           </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem' }}>
           
           {/* Chart 1: Temperature Correlation */}
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>Fever Correlation Matrix</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={mastitisData.temperature} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="name" stroke="var(--text-main)" />
                  <YAxis domain={[35, 50]} stroke="var(--text-main)" label={{ value: 'Core Temp (°C)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                  <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--danger)', color: '#fff' }} />
                  <Bar dataKey="temperature" radius={[8, 8, 0, 0]}>
                    {mastitisData.temperature.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.name === 'Healthy Cows' ? '#4CAF50' : 'var(--danger)'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
           </div>

           {/* Chart 2: Udder Hardness & Pain */}
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>Pain & Hardness Distribution</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={mastitisData.symptoms} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                  <XAxis dataKey="name" stroke="var(--text-main)" />
                  <YAxis stroke="var(--text-main)" label={{ value: 'Percentage of Cows (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                  <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Legend />
                  <Bar dataKey="Severe Pain (%)" fill="#ffeb3b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Udder Hardness (%)" fill="var(--text-highlight)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
           </div>
        </div>

        {/* Chart 3: Radar Chart for Udder Swelling */}
        <div className="glass-panel" style={{ padding: '1.5rem', marginTop: '3rem' }}>
           <h3 style={{color: 'white', marginBottom: '1rem', textAlign: 'center'}}>Physical Udder Swelling (Flex Sensor Limits)</h3>
           <p style={{color: 'var(--text-main)', textAlign: 'center', marginBottom: '1rem'}}>Comparing the 8-point physical expansion limits of Healthy vs Mastitis udders.</p>
           <ResponsiveContainer width="100%" height={400}>
             <RadarChart cx="50%" cy="50%" outerRadius="80%" data={mastitisData.flex_sensors}>
               <PolarGrid stroke="rgba(255,255,255,0.2)" />
               <PolarAngleAxis dataKey="sensor" tick={{ fill: 'var(--text-main)', fontSize: 12 }} />
               <PolarRadiusAxis angle={30} domain={[0, 300]} tick={{ fill: 'var(--text-main)' }} />
               <Radar name="Healthy Udder" dataKey="Healthy Udder" stroke="#4CAF50" fill="#4CAF50" fillOpacity={0.4} />
               <Radar name="Swollen Udder (Mastitis)" dataKey="Swollen Udder (Mastitis)" stroke="var(--danger)" fill="var(--danger)" fillOpacity={0.4} />
               <Legend verticalAlign="bottom" height={36}/>
               <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: 'white' }} />
             </RadarChart>
           </ResponsiveContainer>
        </div>

        {/* Chart 4: 6-Day Progression Timeline */}
        <div className="glass-panel" style={{ padding: '1.5rem', marginTop: '3rem' }}>
           <h3 style={{color: 'white', marginBottom: '1rem'}}>6-Day Symptom Tracking Timeline</h3>
           <p style={{color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: '1rem'}}>Tracking the consistently elevated Fever and Swelling limits over a 6-day period.</p>
           <ResponsiveContainer width="100%" height={320}>
             <ComposedChart data={mastitisData.timeline} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
               <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
               <XAxis dataKey="day" stroke="var(--text-main)" label={{ value: 'Observation Timeline', position: 'insideBottom', offset: -15, fill: 'var(--text-main)' }} height={40} />
               <YAxis yAxisId="left" stroke="#4CAF50" domain={[38, 50]} label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', offset: 15, fill: '#4CAF50' }} width={60} />
               <YAxis yAxisId="right" orientation="right" stroke="var(--danger)" domain={[200, 300]} label={{ value: 'Swelling (ADC Units)', angle: 90, position: 'insideRight', offset: 15, fill: 'var(--danger)' }} width={60} />
               <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: 'white' }} />
               <Legend verticalAlign="top" height={36} />
               <Line yAxisId="left" type="monotone" dataKey="Healthy Temp" stroke="#4CAF50" strokeWidth={3} dot={{r: 4}} />
               <Line yAxisId="left" type="monotone" dataKey="Mastitis Temp" stroke="var(--danger)" strokeWidth={3} strokeDasharray="5 5" dot={{r: 4}} />
               <Line yAxisId="right" type="monotone" dataKey="Healthy Swelling" stroke="#ffeb3b" strokeWidth={2} dot={{r: 4}} />
               <Line yAxisId="right" type="monotone" dataKey="Mastitis Swelling" stroke="var(--text-highlight)" strokeWidth={2} strokeDasharray="5 5" dot={{r: 4}} />
             </ComposedChart>
           </ResponsiveContainer>
        </div>

        {/* ------------------------------------------------ */}
        {/* NEW PHASE 11 ANALYTICS DASHBOARD (7 GRAPHS) */}
        {/* ------------------------------------------------ */}
        <div style={{ marginTop: '4rem', paddingTop: '2rem', borderTop: '2px solid rgba(255,255,255,0.1)' }}>
           <h2 style={{color: 'var(--text-highlight)', marginBottom: '10px'}}>🧬 Deep Statistical Inference</h2>
           <p style={{color: 'var(--text-main)', marginBottom: '3rem'}}>Advanced ML-feature profiling requested by the analytics team.</p>
           
           <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', marginBottom: '3rem' }}>
              
              {/* 1. Mastitis Distribution */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                 <h3 style={{color: 'white', marginBottom: '1rem'}}>1. Dataset Class Distribution</h3>
                 <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Balance check for Machine Learning readiness.</p>
                 <ResponsiveContainer width="100%" height={260}>
                   <BarChart data={mastitisData.distribution} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                     <XAxis dataKey="name" stroke="var(--text-main)" />
                     <YAxis stroke="var(--text-main)" label={{ value: 'Cow Count', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                     <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                     <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                       {mastitisData.distribution.map((entry, index) => (
                         <Cell key={`cell-${index}`} fill={entry.name.includes('Healthy') ? '#4CAF50' : 'var(--danger)'} />
                       ))}
                     </Bar>
                   </BarChart>
                 </ResponsiveContainer>
              </div>

              {/* 2. Temperature Box Plot (Simulated Distribution) */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                 <h3 style={{color: 'white', marginBottom: '1rem'}}>2. Temperature Quartile Profile</h3>
                 <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Comparing Min, Q1, Median, Q3, and Max.</p>
                 <ResponsiveContainer width="100%" height={260}>
                   <BarChart data={mastitisData.temp_box} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                     <XAxis dataKey="name" stroke="var(--text-main)" />
                     <YAxis domain={[20, 60]} stroke="var(--text-main)" label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                     <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                     <Legend />
                     <Bar dataKey="min" fill="rgba(255,255,255,0.2)" name="Minimum" />
                     <Bar dataKey="q1" fill="#ffeb3b" name="25th Pct (Q1)" />
                     <Bar dataKey="median" fill="var(--text-highlight)" name="Median" />
                     <Bar dataKey="q3" fill="var(--danger)" name="75th Pct (Q3)" />
                     <Bar dataKey="max" fill="rgba(207, 102, 121, 0.4)" name="Maximum" />
                   </BarChart>
                 </ResponsiveContainer>
              </div>
           </div>

           {/* 3. KEY GRAPH: Udder Sensor Analysis */}
           <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '3rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem', textAlign: 'center'}}>3. Udder Sensor Quadrant Analysis (KEY GRAPH)</h3>
              <p style={{color: 'var(--text-main)', textAlign: 'center', marginBottom: '1rem'}}>Direct comparison of physical inhale/exhale expansion limits across all 4 udder quarters.</p>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={mastitisData.udder_sensors} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" stroke="var(--text-main)" label={{ value: 'Udder Quarter', position: 'insideBottom', offset: -10, fill: 'var(--text-main)' }} height={40}/>
                  <YAxis domain={[180, 300]} stroke="var(--text-main)" label={{ value: 'Flex Limit (ADC Units)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} width={60} />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Legend verticalAlign="top" height={36}/>
                  <Bar dataKey="Healthy Inhale" fill="#4CAF50" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Healthy Exhale" fill="#81c784" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Mastitis Inhale" fill="var(--danger)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Mastitis Exhale" fill="#e57373" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
           </div>

           <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', marginBottom: '3rem' }}>
              
              {/* 4. Symptoms Stacked */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                 <h3 style={{color: 'white', marginBottom: '1rem'}}>4. Symptom Saturation</h3>
                 <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Stacked percentage of visible physical symptoms.</p>
                 <ResponsiveContainer width="100%" height={280}>
                   <BarChart data={mastitisData.symptoms_stacked} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                     <XAxis dataKey="name" stroke="var(--text-main)" />
                     <YAxis stroke="var(--text-main)" label={{ value: 'Population %', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                     <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                     <Legend />
                     <Bar dataKey="Hardness" stackId="a" fill="var(--danger)" />
                     <Bar dataKey="Pain" stackId="a" fill="#ffeb3b" />
                     <Bar dataKey="Visibility" stackId="a" fill="var(--text-highlight)" name="Abnormal Milk Vis." />
                   </BarChart>
                 </ResponsiveContainer>
              </div>

              {/* 5. Lactation Stage */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                 <h3 style={{color: 'white', marginBottom: '1rem'}}>5. Lactation Stage Impact</h3>
                 <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Mastitis onset rate across the 6-month lactation cycle.</p>
                 <ResponsiveContainer width="100%" height={280}>
                   <AreaChart data={mastitisData.lactation_impact} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                     <XAxis dataKey="months" stroke="var(--text-main)" />
                     <YAxis stroke="var(--text-main)" domain={[20, 60]} label={{ value: 'Mastitis Rate (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                     <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)', color: '#fff' }} />
                     <Area type="monotone" dataKey="rate" stroke="var(--accent-color)" fill="rgba(100, 255, 218, 0.2)" strokeWidth={3} dot={{r: 4}} name="Infection Rate" />
                   </AreaChart>
                 </ResponsiveContainer>
              </div>
           </div>

           <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '3rem', marginBottom: '1rem' }}>
              
              {/* 6. Previous Mastitis Recurrence */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                 <h3 style={{color: 'white', marginBottom: '1rem'}}>6. Recurrence Risk</h3>
                 <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Does prior mastitis predict current infection?</p>
                 <ResponsiveContainer width="100%" height={300}>
                   <BarChart data={mastitisData.recurrence} margin={{ top: 10, right: 10, left: 10, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                     <XAxis dataKey="status" stroke="var(--text-main)" />
                     <YAxis stroke="var(--text-main)" domain={[0, 100]} label={{ value: 'Current Mastitis Rate (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} />
                     <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                     <Bar dataKey="rate" radius={[8, 8, 0, 0]}>
                       {mastitisData.recurrence.map((entry, index) => (
                         <Cell key={`cell-${index}`} fill={entry.rate === 100 ? 'var(--danger)' : '#ffeb3b'} />
                       ))}
                     </Bar>
                   </BarChart>
                 </ResponsiveContainer>
              </div>

              {/* 7. Correlation Heatmap (Bubble Chart) */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                 <h3 style={{color: 'white', marginBottom: '1rem'}}>7. Feature Correlation Mapping</h3>
                 <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Pearson correlation coefficients predicting Mastitis onset (Larger/higher bubbles = stronger signal).</p>
                 <ResponsiveContainer width="100%" height={300}>
                   <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                     <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                     <XAxis dataKey="feature" stroke="var(--text-main)" tick={{ fontSize: 10, angle: -45, textAnchor: 'end' }} height={60} />
                     <YAxis dataKey="correlation" stroke="var(--text-main)" domain={[-0.2, 1.0]} label={{ value: 'Pearson Coefficient', angle: -90, position: 'insideLeft', fill: 'var(--text-main)' }} width={60} />
                     <ZAxis dataKey="z_size" range={[100, 1000]} name="Strength" />
                     <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)', color: '#fff' }} />
                     <Scatter name="Correlation Heat" data={mastitisData.correlation}>
                       {mastitisData.correlation.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.correlation > 0.4 ? 'var(--danger)' : entry.correlation > 0 ? 'var(--text-highlight)' : 'rgba(255,255,255,0.3)'} />
                       ))}
                     </Scatter>
                   </ScatterChart>
                 </ResponsiveContainer>
              </div>
           </div>
        </div>

      </div>
    );
  };

  // ------------ TAB 6: GLOBAL EPIDEMIOLOGY ------------
  const renderEpidemiologyTab = () => {
    if (!epidemiologyData) return (
      <div style={{color:'white', padding:'3rem', textAlign:'center'}}>
        <div style={{fontSize:'2rem', marginBottom:'1rem'}}>Loading...</div>
        <p style={{color:'var(--text-main)'}}>Aggregating 250,000 Global Cattle Records...</p>
      </div>
    );
    return (
      <div style={{ animation: 'fade-in 0.5s' }}>
        <div style={{ marginBottom: '2.5rem' }}>
          <h2 style={{color: 'white', fontSize: '2.2rem', marginBottom: '10px'}}>Global Epidemiology Dashboard</h2>
          <p style={{color: 'var(--text-main)', fontSize: '1.1rem'}}>Multi-variate analysis of 250,000 global cattle disease records across climates, diets, and vaccination profiles.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', marginBottom: '3rem' }}>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>1. Global Disease Distribution</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Top 15 most prevalent cattle diseases globally.</p>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart layout="vertical" data={epidemiologyData.distribution} margin={{ top: 10, right: 30, left: 100, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                  <XAxis type="number" stroke="var(--text-main)" />
                  <YAxis type="category" dataKey="name" stroke="var(--text-main)" tick={{fontSize: 11, fill: 'var(--text-main)'}} width={120} />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Bar dataKey="count" fill="var(--danger)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
           </div>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>2. Physiological Vitals Index</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Core Body Temp vs Heart Rate across top infections.</p>
              <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={epidemiologyData.vitals} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="disease" stroke="var(--text-main)" tick={{fontSize: 10, angle: -45, textAnchor: 'end'}} height={60} />
                  <YAxis yAxisId="temp" domain={[38, 39]} stroke="var(--text-highlight)" />
                  <YAxis yAxisId="hr" orientation="right" domain={[60, 70]} stroke="var(--danger)" />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Legend verticalAlign="top" />
                  <Bar yAxisId="temp" dataKey="Temperature" fill="var(--text-highlight)" radius={[4, 4, 0, 0]} />
                  <Line yAxisId="hr" type="monotone" dataKey="HeartRate" stroke="var(--danger)" strokeWidth={3} dot={{r: 4}} />
                </ComposedChart>
              </ResponsiveContainer>
           </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem', marginBottom: '3rem' }}>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>3. Environmental Vulnerability</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Seasonal disease rates across distinct global climates.</p>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={epidemiologyData.environment} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="Climate" stroke="var(--text-main)" />
                  <YAxis stroke="var(--text-main)" />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Legend />
                  <Bar dataKey="Summer" stackId="a" fill="var(--danger)" />
                  <Bar dataKey="Monsoon" stackId="a" fill="#2196F3" />
                  <Bar dataKey="Winter" stackId="a" fill="#9C27B0" />
                  <Bar dataKey="Spring" stackId="a" fill="#4CAF50" />
                  <Bar dataKey="Autumn" stackId="a" fill="#FF9800" />
                </BarChart>
              </ResponsiveContainer>
           </div>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>4. Nutritional Deficiencies</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Overall herd illness rate mapped against Feed Type.</p>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={epidemiologyData.nutrition}>
                  <PolarGrid stroke="rgba(255,255,255,0.2)" />
                  <PolarAngleAxis dataKey="feed" tick={{ fill: 'var(--text-main)', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[40, 50]} tick={{ fill: 'var(--text-main)' }} />
                  <Radar name="Disease Rate (%)" dataKey="Disease Rate" stroke="var(--text-highlight)" fill="var(--text-highlight)" fillOpacity={0.4} />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: 'white' }} />
                </RadarChart>
              </ResponsiveContainer>
           </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '2rem', marginBottom: '3rem' }}>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>5. Vaccination Efficacy</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Infection rate drop: Foot and Mouth Disease.</p>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={epidemiologyData.vaccination} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="status" stroke="var(--text-main)" />
                  <YAxis stroke="var(--text-main)" domain={[0, 2]} />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Bar dataKey="FMD Infection Rate" radius={[4, 4, 0, 0]}>
                     {epidemiologyData.vaccination.map((entry, index) => (
                       <Cell key={`cell-${index}`} fill={entry.status === 'Vaccinated' ? '#4CAF50' : 'var(--danger)'} />
                     ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
           </div>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>6. Behavioral Shifts</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Healthy vs Sick Activity Levels.</p>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={epidemiologyData.activity} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="status" stroke="var(--text-main)" />
                  <YAxis stroke="var(--text-main)" />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--text-main)', color: '#fff' }} />
                  <Legend />
                  <Bar dataKey="Walking (km)" fill="var(--text-highlight)" />
                  <Bar dataKey="Resting (hrs)" fill="#9C27B0" />
                </BarChart>
              </ResponsiveContainer>
           </div>
           <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{color: 'white', marginBottom: '1rem'}}>7. Disease Correlation Map</h3>
              <p style={{color: 'var(--text-main)', fontSize: '0.85rem'}}>Pearson strengths predicting overall sickness.</p>
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="feature" stroke="var(--text-main)" tick={{fontSize: 9, angle: -45, textAnchor: 'end'}} height={50} />
                  <YAxis dataKey="correlation" stroke="var(--text-main)" domain={[-0.1, 0.1]} width={40} />
                  <ZAxis dataKey="z_size" range={[100, 600]} name="Strength" />
                  <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--accent-color)', color: '#fff' }} />
                  <Scatter name="Correlation Heat" data={epidemiologyData.correlation} fill="var(--danger)" />
                </ScatterChart>
              </ResponsiveContainer>
           </div>
        </div>
      </div>
    );
  };

  // ------------ TAB 5: AI SMART FARM SIMULATOR (XGBOOST) ------------
  // ------------ TAB 5: AI SMART FARM SIMULATOR (XGBOOST) ------------
  const renderSmartSimulatorTab = () => {
    return (
      <div style={{ animation: 'fade-in 0.8s' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h2 style={{color: 'white', fontSize: '2rem', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '15px'}}>
            <Activity color="#ff9800" size={32} /> Cattle Health Predictor
          </h2>
          <p style={{color: 'var(--text-main)'}}>Dual XGBoost engine: Disease Risk + Milk Yield Intelligence in one unified inference pass.</p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: '2rem' }}>
          {/* CONTROL PANEL */}
          <div className="glass-panel" style={{ padding: '1.5rem', height: 'fit-content' }}>
            <h3 style={{color: 'var(--text-highlight)', marginBottom: '1.5rem', fontSize: '1rem', letterSpacing: '1px'}}>FARM CONTROL KNOBS</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-highlight)', fontWeight: 'bold', letterSpacing: '2px' }}>BIOLOGICAL VITALS</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Body Temp (C)</label>
                  <input type="number" step="0.1" value={smartInputs.body_temp}
                    onChange={(e) => setSmartInputs({...smartInputs, body_temp: parseFloat(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Heart Rate (BPM)</label>
                  <input type="number" value={smartInputs.heart_rate}
                    onChange={(e) => setSmartInputs({...smartInputs, heart_rate: parseInt(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Resp. Rate (/min)</label>
                  <input type="number" value={smartInputs.resp_rate}
                    onChange={(e) => setSmartInputs({...smartInputs, resp_rate: parseInt(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-highlight)', fontWeight: 'bold', letterSpacing: '2px' }}>ENVIRONMENT</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Ambient Temp (C)</label>
                  <input type="number" value={smartInputs.ambient_temp}
                    onChange={(e) => setSmartInputs({...smartInputs, ambient_temp: parseFloat(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Humidity (%)</label>
                  <input type="number" value={smartInputs.humidity}
                    onChange={(e) => setSmartInputs({...smartInputs, humidity: parseFloat(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-highlight)', fontWeight: 'bold', letterSpacing: '2px' }}>NUTRITION + LACTATION</div>
              <div>
                <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Feed Quantity: {smartInputs.feed_qty} kg</label>
                <input type="range" min="5" max="30" step="0.5" value={smartInputs.feed_qty}
                  onChange={(e) => setSmartInputs({...smartInputs, feed_qty: parseFloat(e.target.value)})}
                  style={{width: '100%', accentColor: 'var(--text-highlight)'}} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Feeding Freq/day</label>
                  <input type="number" min="1" max="6" value={smartInputs.feeding_frequency}
                    onChange={(e) => setSmartInputs({...smartInputs, feeding_frequency: parseInt(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Days in Milk</label>
                  <input type="number" value={smartInputs.days_in_milk}
                    onChange={(e) => setSmartInputs({...smartInputs, days_in_milk: parseInt(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Prev Week Avg (L)</label>
                  <input type="number" step="0.1" value={smartInputs.prev_week_avg}
                    onChange={(e) => setSmartInputs({...smartInputs, prev_week_avg: parseFloat(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
                <div>
                  <label style={{color: 'var(--text-main)', display: 'block', marginBottom: '4px', fontSize: '0.75rem'}}>Actual Yield Today (L)</label>
                  <input type="number" step="0.1" value={smartInputs.actual_yield}
                    onChange={(e) => setSmartInputs({...smartInputs, actual_yield: parseFloat(e.target.value)})}
                    style={{width: '100%', padding: '7px', borderRadius: '4px', background: 'rgba(0,0,0,0.4)', color: 'white', border: '1px solid rgba(255,255,255,0.15)', boxSizing: 'border-box'}} />
                </div>
              </div>

              <button onClick={handleSmartInference} disabled={isInferringSmart}
                style={{ padding: '14px', borderRadius: '8px', border: 'none',
                  background: isInferringSmart ? 'rgba(100,255,218,0.3)' : 'var(--text-highlight)',
                  color: 'black', fontWeight: 'bold', fontSize: '1rem',
                  cursor: isInferringSmart ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                {isInferringSmart ? 'Checking...' : <><Zap size={18}/> Check Cow Health & Yield</>}
              </button>
            </div>
          </div>

          {/* OUTPUT PANEL */}
          <div>
            {!smartResult && !isInferringSmart && (
              <div style={{ height: '450px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', border: '2px dashed rgba(255,255,255,0.08)', borderRadius: '16px' }}>
                <BrainCircuit size={70} style={{opacity: 0.15, marginBottom: '16px'}}/>
                <p style={{color: 'var(--text-main)', fontSize: '1.1rem'}}>Enter the cow's vitals and environment on the left to check her health.</p>
                <p style={{color: 'var(--text-main)', fontSize: '0.85rem', marginTop: '6px'}}>The AI will automatically predict both disease risk and milk yield.</p>
              </div>
            )}
            {isInferringSmart && (
              <div style={{ height: '450px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <div style={{ width: '48px', height: '48px', border: '4px solid rgba(100,255,218,0.1)', borderTopColor: 'var(--text-highlight)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}></div>
                <p style={{color: 'var(--text-highlight)', marginTop: '18px', fontSize: '1rem'}}>Analyzing cow's vitals...</p>
              </div>
            )}
            {smartResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                {/* ROW 1: Disease + Yield side by side */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

                  {/* DISEASE CARD */}
                  <div className="glass-panel" style={{ padding: '1.5rem', borderTop: smartResult.health.risk_pct > 50 ? '4px solid var(--danger)' : smartResult.health.risk_pct > 30 ? '4px solid #FF9800' : '4px solid #4CAF50' }}>
                    <h3 style={{color: 'white', marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem'}}><HeartPulse size={20}/> Cow Health Status</h3>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '1.2rem' }}>
                      <div style={{ width: '100px', height: '100px', flexShrink: 0 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <RadialBarChart cx="50%" cy="50%" innerRadius="65%" outerRadius="100%" barSize={10}
                            data={[{ name: 'Risk', value: smartResult.health.risk_pct, fill: smartResult.health.risk_pct > 50 ? 'var(--danger)' : smartResult.health.risk_pct > 30 ? '#FF9800' : '#4CAF50' }]}
                            startAngle={180} endAngle={0}>
                            <RadialBar background dataKey="value" cornerRadius={8} />
                            <text x="50%" y="42%" textAnchor="middle" dominantBaseline="middle" style={{ fill: 'white', fontSize: '1rem', fontWeight: 'bold' }}>
                              {smartResult.health.risk_pct}%
                            </text>
                          </RadialBarChart>
                        </ResponsiveContainer>
                      </div>
                      <div>
                        <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: smartResult.health.risk_pct > 50 ? 'var(--danger)' : smartResult.health.risk_pct > 30 ? '#FF9800' : '#4CAF50' }}>{smartResult.health.status}</div>
                        <div style={{ color: 'var(--text-main)', fontSize: '0.8rem', marginTop: '4px' }}>Sickness Probability</div>
                      </div>
                    </div>
                    {smartResult.health.warnings.length > 0 && (
                      <div style={{ marginBottom: '1rem' }}>
                        {smartResult.health.warnings.map((w, i) => (
                          <div key={i} style={{ padding: '6px 10px', background: 'rgba(255,82,82,0.1)', borderLeft: '3px solid var(--danger)', borderRadius: '4px', color: '#ffb3b3', fontSize: '0.8rem', marginBottom: '5px' }}>
                            {w}
                          </div>
                        ))}
                      </div>
                    )}
                    <div style={{ background: 'rgba(255,255,255,0.04)', padding: '1rem', borderRadius: '8px', fontSize: '0.85rem', color: 'white', lineHeight: '1.5' }}>
                      {smartResult.health.decision}
                    </div>
                  </div>

                  {/* YIELD CARD */}
                  <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid var(--text-highlight)' }}>
                    <h3 style={{color: 'white', marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem'}}><BarChart3 size={20}/> Expected Milk Yield</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem', marginBottom: '1rem' }}>
                      <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginBottom: '4px' }}>Predicted Yield</div>
                        <div style={{ color: 'var(--text-highlight)', fontSize: '1.6rem', fontWeight: 'bold' }}>{smartResult.production.predicted_yield}L</div>
                      </div>
                      <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.9rem', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginBottom: '4px' }}>Performance Score</div>
                        <div style={{ color: '#ffeb3b', fontSize: '1.6rem', fontWeight: 'bold' }}>{smartResult.production.performance_score}/100</div>
                      </div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.04)', padding: '0.9rem', borderRadius: '8px', textAlign: 'center', marginBottom: '0.8rem' }}>
                      <div style={{ color: 'var(--text-main)', fontSize: '0.75rem' }}>Cow Performance Label</div>
                      <div style={{ color: '#ffeb3b', fontWeight: 'bold', fontSize: '1rem', marginTop: '4px' }}>{smartResult.production.performance_label}</div>
                    </div>
                    {smartResult.production.yield_alert && (
                      <div style={{ padding: '10px', background: 'rgba(255,82,82,0.12)', border: '1px solid rgba(255,82,82,0.4)', borderRadius: '6px', color: '#ffb3b3', fontSize: '0.8rem' }}>
                        {smartResult.production.yield_alert}
                      </div>
                    )}
                  </div>
                </div>

                {/* ROW 2: 7 Intelligence Metrics */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                  <div className="glass-panel" style={{ padding: '1.2rem', borderLeft: '3px solid #4CAF50' }}>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginBottom: '4px' }}>Feed +2kg Gain</div>
                    <div style={{ color: '#4CAF50', fontSize: '1.4rem', fontWeight: 'bold' }}>+{smartResult.production.feed_gain}L</div>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginTop: '4px' }}>per day yield boost</div>
                  </div>
                  <div className="glass-panel" style={{ padding: '1.2rem', borderLeft: '3px solid var(--danger)' }}>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginBottom: '4px' }}>Heat Stress Loss</div>
                    <div style={{ color: 'var(--danger)', fontSize: '1.4rem', fontWeight: 'bold' }}>-{smartResult.production.climate_penalty}L</div>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginTop: '4px' }}>vs ideal 22C/50%RH</div>
                  </div>
                  <div className="glass-panel" style={{ padding: '1.2rem', borderLeft: `3px solid ${smartResult.production.trend_delta >= 0 ? '#4CAF50' : 'var(--danger)'}` }}>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginBottom: '4px' }}>7-Day Yield Trend</div>
                    <div style={{ color: smartResult.production.trend_delta >= 0 ? '#4CAF50' : 'var(--danger)', fontSize: '1.4rem', fontWeight: 'bold' }}>
                      {smartResult.production.trend_delta >= 0 ? '+' : ''}{smartResult.production.trend_delta}L
                    </div>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginTop: '4px' }}>Next week: {smartResult.production.future_yield_7d}L</div>
                  </div>
                  <div className="glass-panel" style={{ padding: '1.2rem', borderLeft: '3px solid var(--text-highlight)' }}>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginBottom: '4px' }}>What-If: Feed +5kg</div>
                    <div style={{ color: 'var(--text-highlight)', fontSize: '1.4rem', fontWeight: 'bold' }}>{smartResult.production.whatif_yield_plus5kg}L</div>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.7rem', marginTop: '4px' }}>max scenario yield</div>
                  </div>
                </div>

                {/* ROW 3: Yield bar chart */}
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h3 style={{ color: 'white', marginBottom: '1rem', fontSize: '0.95rem' }}>Yield Scenario Comparison</h3>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={[
                      { name: 'Actual', value: smartInputs.actual_yield, fill: '#9E9E9E' },
                      { name: 'AI Predicted', value: smartResult.production.predicted_yield, fill: 'var(--text-highlight)' },
                      { name: 'Feed Optimized', value: smartResult.production.optimized_yield, fill: '#4CAF50' },
                      { name: 'What-If +5kg', value: smartResult.production.whatif_yield_plus5kg, fill: '#2196F3' },
                      { name: 'Ideal Climate', value: parseFloat((smartResult.production.predicted_yield + smartResult.production.climate_penalty).toFixed(2)), fill: '#FF9800' },
                    ]} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                      <XAxis dataKey="name" stroke="var(--text-main)" tick={{ fontSize: 11 }} />
                      <YAxis stroke="var(--text-main)" domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', borderColor: 'var(--text-highlight)', color: '#fff' }} formatter={(v) => [v + ' L', 'Yield']} />
                      <Bar dataKey="value" radius={[4,4,0,0]}>
                        {[
                          { fill: '#9E9E9E' },
                          { fill: 'var(--text-highlight)' },
                          { fill: '#4CAF50' },
                          { fill: '#2196F3' },
                          { fill: '#FF9800' },
                        ].map((c, i) => <Cell key={i} fill={c.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* ROW 4: What-If Disease Impact Simulator */}
                <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid var(--danger)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ color: 'white', margin: 0, display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}><TrendingDown size={20}/> What-If Disease Impact Simulator</h3>
                    <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                      <select 
                        value={impactDisease}
                        onChange={(e) => setImpactDisease(e.target.value)}
                        style={{ padding: '8px 12px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '6px', outline: 'none' }}>
                        {diseaseOptions.filter(d => d !== 'Healthy').map((d, i) => <option key={i} value={d}>{d.replace(/_/g, ' ')}</option>)}
                      </select>
                      <button onClick={handleImpactInference} disabled={isInferringImpact}
                        style={{ padding: '8px 16px', borderRadius: '6px', border: 'none', background: 'var(--danger)', color: 'white', fontWeight: 'bold', cursor: isInferringImpact ? 'not-allowed' : 'pointer' }}>
                        {isInferringImpact ? 'Calculating...' : 'Simulate Drop'}
                      </button>
                    </div>
                  </div>

                  {impactResult && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
                      <div style={{ background: 'rgba(76, 175, 80, 0.1)', border: '1px solid #4CAF50', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ color: '#4CAF50', fontSize: '0.85rem', marginBottom: '8px' }}>Healthy Yield Baseline</div>
                        <div style={{ color: 'white', fontSize: '1.8rem', fontWeight: 'bold' }}>{impactResult.healthy_yield}L</div>
                      </div>
                      <div style={{ background: 'rgba(255, 152, 0, 0.1)', border: '1px solid #FF9800', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ color: '#FF9800', fontSize: '0.85rem', marginBottom: '8px' }}>Yield if sick with {impactDisease.replace(/_/g, ' ')}</div>
                        <div style={{ color: 'white', fontSize: '1.8rem', fontWeight: 'bold' }}>{impactResult.sick_yield}L</div>
                      </div>
                      <div style={{ background: 'rgba(255, 82, 82, 0.1)', border: '1px solid var(--danger)', padding: '1rem', borderRadius: '8px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--danger)', fontSize: '0.85rem', marginBottom: '8px' }}>Total Exact Loss</div>
                        <div style={{ color: 'var(--danger)', fontSize: '1.8rem', fontWeight: 'bold' }}>{impactResult.loss}L</div>
                        <div style={{ color: '#ffb3b3', fontSize: '0.75rem', marginTop: '4px' }}>per day while sick</div>
                      </div>
                    </div>
                  )}
                  {!impactResult && !isInferringImpact && (
                    <div style={{ textAlign: 'center', color: 'var(--text-main)', padding: '1rem 0' }}>
                      Select a disease from the dropdown and click "Simulate Drop" to predict the exact liter loss.
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>
        </div>
      </div>
    );
  };


  // ------------ NEW: SYMPTOM CHECKER TAB ------------
  const renderSymptomCheckerTab = () => {
    return (
      <div style={{ padding: '0 1rem', animation: 'fade-in 0.5s' }}>
        <h2 style={{ color: 'white', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity color="#FF9800" /> Cow Symptom Checker
        </h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          {/* Controls */}
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <h3 style={{ color: 'var(--text-highlight)', marginBottom: '1.5rem' }}>Enter the cow's vitals</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Body Temperature (°C)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="36.0" max="42.0" step="0.1" value={symptomInputs.body_temp} onChange={(e) => setSymptomInputs({...symptomInputs, body_temp: parseFloat(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.body_temp}°C</span>
                </div>
              </div>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Heart Rate (BPM)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="40" max="140" step="1" value={symptomInputs.heart_rate} onChange={(e) => setSymptomInputs({...symptomInputs, heart_rate: parseInt(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.heart_rate}</span>
                </div>
              </div>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Respiratory Rate</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="10" max="80" step="1" value={symptomInputs.resp_rate} onChange={(e) => setSymptomInputs({...symptomInputs, resp_rate: parseInt(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.resp_rate}</span>
                </div>
              </div>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Daily Feed Intake (kg)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="0" max="30" step="0.5" value={symptomInputs.feed_qty} onChange={(e) => setSymptomInputs({...symptomInputs, feed_qty: parseFloat(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.feed_qty}kg</span>
                </div>
              </div>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Current Milk Yield (L)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="0" max="40" step="0.5" value={symptomInputs.milk_yield} onChange={(e) => setSymptomInputs({...symptomInputs, milk_yield: parseFloat(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.milk_yield}L</span>
                </div>
              </div>
            </div>

            <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', marginTop: '1.5rem', paddingTop: '1.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Ambient Temp (°C)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="-10" max="45" step="1" value={symptomInputs.ambient_temp} onChange={(e) => setSymptomInputs({...symptomInputs, ambient_temp: parseFloat(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.ambient_temp}°C</span>
                </div>
              </div>
              <div>
                <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Humidity (%)</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <input type="range" min="10" max="100" step="1" value={symptomInputs.humidity} onChange={(e) => setSymptomInputs({...symptomInputs, humidity: parseFloat(e.target.value)})} style={{flex:1}} />
                  <span style={{color: 'white', minWidth: '40px'}}>{symptomInputs.humidity}%</span>
                </div>
              </div>
            </div>
            
            <button 
              onClick={handleSymptomInference}
              disabled={isInferringSymptoms}
              style={{ width: '100%', padding: '1rem', marginTop: '2rem', backgroundColor: '#FF9800', color: 'black', border: 'none', borderRadius: '8px', fontSize: '1.1rem', fontWeight: 'bold', cursor: 'pointer' }}>
              {isInferringSymptoms ? 'Analyzing Symptoms...' : 'Check Symptoms'}
            </button>
          </div>

          {/* Results */}
          <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid #FF9800' }}>
            <h3 style={{ color: 'white', marginBottom: '1.5rem' }}>AI Diagnosis</h3>
            {symptomResult ? (
              symptomResult.error ? (
                <div style={{ padding: '1.5rem', backgroundColor: 'rgba(255, 82, 82, 0.1)', border: '1px solid var(--danger)', borderRadius: '8px', color: '#ffb3b3' }}>
                  <h4 style={{ margin: '0 0 10px 0' }}>Error connecting to AI Engine</h4>
                  <p>{symptomResult.error}</p>
                  <p style={{ fontSize: '0.85rem', marginTop: '10px' }}>Make sure the FastAPI backend is running and the new endpoints are loaded.</p>
                </div>
              ) : (
                <div>
                  <div style={{ padding: '1.5rem', backgroundColor: 'rgba(255, 152, 0, 0.1)', border: '1px solid #FF9800', borderRadius: '8px', marginBottom: '1.5rem' }}>
                    <h4 style={{ color: 'var(--text-main)', margin: '0 0 10px 0', fontSize: '0.9rem' }}>What the AI found:</h4>
                    <div style={{ color: '#FF9800', fontSize: '2rem', fontWeight: 'bold', marginBottom: '10px' }}>
                      {symptomResult.primary_diagnosis}
                    </div>
                    <div style={{ color: 'white', fontSize: '0.95rem', lineHeight: '1.4' }}>
                      {symptomResult.reasoning}
                    </div>
                  </div>

                  <h4 style={{ color: 'var(--text-main)', marginBottom: '1rem', fontSize: '0.9rem' }}>Top 3 Probable Conditions:</h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {symptomResult.top_predictions.map((pred, idx) => (
                      <div key={idx}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                          <span style={{ color: 'white', fontWeight: idx === 0 ? 'bold' : 'normal' }}>{pred.disease}</span>
                          <span style={{ color: idx === 0 ? '#FF9800' : 'var(--text-main)' }}>{pred.probability}%</span>
                        </div>
                        <div style={{ height: '8px', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                          <div style={{ width: `${pred.probability}%`, height: '100%', backgroundColor: idx === 0 ? '#FF9800' : 'var(--text-main)' }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )
            ) : (
              <div style={{ color: 'var(--text-main)', textAlign: 'center', paddingTop: '3rem' }}>
                <Stethoscope size={48} color="rgba(255,255,255,0.2)" style={{marginBottom:'1rem'}}/>
                <p>Adjust the vitals on the left and run diagnostics to see the exact disease prediction.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // ------------ NEW: FUTURE LACTATION PREDICTOR TAB ------------
  const renderFutureLactationTab = () => {
    return (
      <div style={{ padding: '0 1rem', animation: 'fade-in 0.5s' }}>
        <h2 style={{ color: 'white', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <TrendingUp size={28} color="#FF9800" /> Future Lactation Predictor
        </h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          
          {/* COLUMN 1: PEAK YIELD PREDICTOR */}
          <div>
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h3 style={{ color: 'var(--text-highlight)', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}><Milk size={20}/> How much milk will she produce?</h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Upcoming Lactation Number (e.g., 2nd)</label>
                  <input type="number" value={peakInputs.lactation_number} onChange={(e) => setPeakInputs({...peakInputs, lactation_number: parseInt(e.target.value)})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }} />
                </div>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Length of Previous Lactation (Days)</label>
                  <input type="number" value={peakInputs.length_of_lactation} onChange={(e) => setPeakInputs({...peakInputs, length_of_lactation: parseInt(e.target.value)})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }} />
                </div>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Current Days Dry (Rest Period)</label>
                  <input type="number" value={peakInputs.days_dry} onChange={(e) => setPeakInputs({...peakInputs, days_dry: parseInt(e.target.value)})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }} />
                </div>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Previous Total Yield (Liters)</label>
                  <input type="number" value={peakInputs.total_milk_yield} onChange={(e) => setPeakInputs({...peakInputs, total_milk_yield: parseFloat(e.target.value)})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }} />
                </div>
              </div>
              <button onClick={handlePeakPredict} disabled={isInferringPeak} style={{ width: '100%', padding: '12px', marginTop: '1.5rem', background: 'var(--text-highlight)', color: 'black', fontWeight: 'bold', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                {isInferringPeak ? 'Checking records...' : 'Calculate Future Yield'}
              </button>
            </div>

            {peakResult && (
              <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid var(--text-highlight)' }}>
                <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
                  <div style={{ color: 'var(--text-main)', fontSize: '0.9rem' }}>Predicted Next Peak</div>
                  <div style={{ color: 'var(--text-highlight)', fontSize: '2.5rem', fontWeight: 'bold' }}>{peakResult.predicted_peak_yield} L</div>
                </div>
                <div style={{ background: 'rgba(255,255,255,0.05)', padding: '1rem', borderRadius: '6px' }}>
                  <div style={{ color: peakResult.color, fontWeight: 'bold', marginBottom: '5px' }}>{peakResult.category}</div>
                  <div style={{ color: 'white', fontSize: '0.9rem', lineHeight: '1.4' }}>{peakResult.recommendation}</div>
                </div>
              </div>
            )}
          </div>

          {/* COLUMN 2: QUALITY PREDICTOR */}
          <div>
            <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
              <h3 style={{ color: '#FF9800', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '10px' }}><Droplets size={20}/> How rich will the milk be?</h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Breed</label>
                  <select value={qualityInputs.breed} onChange={(e) => setQualityInputs({...qualityInputs, breed: e.target.value})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }}>
                    <option value="Holstein">Holstein (High Volume)</option>
                    <option value="Czech Fleckvieh">Czech Fleckvieh (Dual Purpose)</option>
                    <option value="Crossbreed">Crossbreed</option>
                  </select>
                </div>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Parity (Number of Calvings)</label>
                  <input type="number" min="1" max="8" value={qualityInputs.parity} onChange={(e) => setQualityInputs({...qualityInputs, parity: parseInt(e.target.value)})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }} />
                </div>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Expected Calving Season</label>
                  <select value={qualityInputs.calving_season} onChange={(e) => setQualityInputs({...qualityInputs, calving_season: e.target.value})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }}>
                    <option value="Spring">Spring</option>
                    <option value="Summer">Summer</option>
                    <option value="Autumn">Autumn</option>
                    <option value="Winter">Winter</option>
                  </select>
                </div>
                <div>
                  <label style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Calving Interval (Days)</label>
                  <input type="number" value={qualityInputs.calv_int} onChange={(e) => setQualityInputs({...qualityInputs, calv_int: parseInt(e.target.value)})} style={{ width: '100%', padding: '8px', background: 'rgba(0,0,0,0.5)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '4px' }} />
                </div>
              </div>
              <button onClick={handleQualityPredict} disabled={isInferringQuality} style={{ width: '100%', padding: '12px', marginTop: '1.5rem', background: '#FF9800', color: 'black', fontWeight: 'bold', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                {isInferringQuality ? 'Analyzing genetics...' : 'Check Milk Quality'}
              </button>
            </div>

            {qualityResult && (
              <div className="glass-panel" style={{ padding: '1.5rem', borderTop: '4px solid #FF9800' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                  <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '6px' }}>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Predicted Fat</div>
                    <div style={{ color: 'white', fontSize: '2rem', fontWeight: 'bold' }}>{qualityResult.predicted_fat_pct}%</div>
                  </div>
                  <div style={{ textAlign: 'center', padding: '1rem', background: 'rgba(255,255,255,0.05)', borderRadius: '6px' }}>
                    <div style={{ color: 'var(--text-main)', fontSize: '0.85rem' }}>Predicted Protein</div>
                    <div style={{ color: 'white', fontSize: '2rem', fontWeight: 'bold' }}>{qualityResult.predicted_protein_pct}%</div>
                  </div>
                </div>
                <div style={{ textAlign: 'center', color: qualityResult.color, fontWeight: 'bold', fontSize: '1.1rem' }}>
                  {qualityResult.classification}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // ------------ SIDEBAR HELPER ------------
  const renderSidebarItem = (id, icon, label, themeColor = 'var(--text-highlight)') => {
    const isActive = activeTab === id;
    const isOrange = themeColor === '#ff9800';
    return (
      <button 
        onClick={() => setActiveTab(id)}
        style={{ 
          display: 'flex', alignItems: 'center', gap: '12px', padding: '10px 16px', margin: '2px 8px',
          backgroundColor: isActive ? (isOrange ? 'rgba(255,152,0, 0.1)' : 'rgba(100,255,218, 0.1)') : 'transparent', 
          color: isActive ? themeColor : 'var(--text-main)', 
          border: 'none', 
          borderLeft: isActive ? `3px solid ${themeColor}` : '3px solid transparent', 
          borderRadius: '4px',
          cursor: 'pointer', fontSize: '0.95rem', fontWeight: isActive ? 'bold' : 'normal', transition: '0.2s',
          whiteSpace: 'nowrap', textAlign: 'left'
        }}>
        <span style={{ fontSize: '1.1rem' }}>{icon}</span>
        {label}
      </button>
    );
  };

  // ------------ MAIN APP RETURN ------------
  return (
    <div className="dashboard-container" style={{ display: 'flex', flexDirection: 'row', height: '100vh', overflow: 'hidden' }}>
      
      {/* Sidebar */}
      <div style={{ 
        width: isSidebarOpen ? '280px' : '65px', 
        transition: 'width 0.3s ease',
        backgroundColor: '#0b0c10', 
        borderRight: '1px solid rgba(255,255,255,0.1)',
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
        overflowX: 'hidden'
      }}>
        {/* Header */}
        <div style={{ padding: '1.2rem 1rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', opacity: isSidebarOpen ? 1 : 0, transition: 'opacity 0.2s', whiteSpace: 'nowrap', overflow: 'hidden' }}>
            <Milk size={24} color="var(--text-highlight)" style={{ flexShrink: 0 }} />
            <h1 style={{ color: 'white', margin: 0, fontSize: '1.1rem', letterSpacing: '0.5px' }}>Ag-AI Platform</h1>
          </div>
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} style={{ background: 'transparent', border: 'none', color: 'var(--text-main)', cursor: 'pointer', padding: '5px' }}>
            <Menu size={20} />
          </button>
        </div>

        {/* Sidebar Links */}
        <div style={{ display: 'flex', flexDirection: 'column', padding: '1rem 0', opacity: isSidebarOpen ? 1 : 0, transition: 'opacity 0.2s', flex: 1, pointerEvents: isSidebarOpen ? 'auto' : 'none' }}>
          
          {/* Analytics Group */}
          <div style={{ marginBottom: '1rem' }}>
            <div 
              onClick={() => setOpenSections({...openSections, analytics: !openSections.analytics})}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.8rem 1.2rem', cursor: 'pointer', color: 'var(--text-main)' }}>
              <div style={{ fontSize: '0.75rem', letterSpacing: '1px', fontWeight: 'bold', textTransform: 'uppercase' }}>📊 Past Records Analysis</div>
              {openSections.analytics ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </div>
            {openSections.analytics && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {renderSidebarItem('farm', '🐄', 'Recent Sensor Logs')}
                {renderSidebarItem('research', '🔬', 'Past Performance Data')}
                {renderSidebarItem('epidemiology', '🌍', 'Disease Pattern Reports')}
              </div>
            )}
          </div>

          <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', margin: '0 1.2rem 1rem 1.2rem' }}></div>

          {/* AI Models Group */}
          <div>
            <div 
              onClick={() => setOpenSections({...openSections, aiModels: !openSections.aiModels})}
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.8rem 1.2rem', cursor: 'pointer', color: '#ff9800' }}>
              <div style={{ fontSize: '0.75rem', letterSpacing: '1px', fontWeight: 'bold', textTransform: 'uppercase' }}>🤖 Health & Predictive Models</div>
              {openSections.aiModels ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </div>
            {openSections.aiModels && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {renderSidebarItem('ai_smart_farm', '🤖', 'Cattle Health Predictor', '#ff9800')}
                {renderSidebarItem('future_lactation', '📈', 'Milk Yield Forecaster', '#ff9800')}
                {renderSidebarItem('vision', '👁️', 'Visual Yield Predictor', '#ff9800')}
                {renderSidebarItem('mastitis', '🩺', 'Mastitis Early Warning', '#ff9800')}
                {renderSidebarItem('symptom_checker', '⚕️', "Cow Symptom Checker", '#ff9800')}
              </div>
            )}
          </div>

        </div>
      </div>

      {/* Main Tab Viewport */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', backgroundColor: 'var(--bg-color)' }}>
         {activeTab === 'farm' && renderFarmTab()}
         {activeTab === 'research' && renderResearchTab()}
         {activeTab === 'vision' && renderVisionTab()}
         {activeTab === 'mastitis' && renderMastitisTab()}
         {activeTab === 'epidemiology' && renderEpidemiologyTab()}
         {activeTab === 'ai_smart_farm' && renderSmartSimulatorTab()}
         {activeTab === 'symptom_checker' && renderSymptomCheckerTab()}
         {activeTab === 'future_lactation' && renderFutureLactationTab()}
      </div>

    </div>
  );
}

export default App;
