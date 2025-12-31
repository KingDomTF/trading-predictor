import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, Activity, Target, Shield, Brain, Zap, Radio } from 'lucide-react';

const TitanOracleTerminal = () => {
  const [selectedAsset, setSelectedAsset] = useState('XAUUSD');
  const [timeframe, setTimeframe] = useState('H1');
  const [liveData, setLiveData] = useState(null);
  const [marketRegime, setMarketRegime] = useState('RANGING');
  const [isLive, setIsLive] = useState(true);

  const assets = [
    { symbol: 'XAUUSD', name: 'Gold', correlation: 0.85, volatility: 1.2 },
    { symbol: 'BTCUSD', name: 'Bitcoin', correlation: -0.3, volatility: 3.8 },
    { symbol: 'US500', name: 'S&P 500', correlation: 0.62, volatility: 0.9 },
    { symbol: 'ETHUSD', name: 'Ethereum', correlation: -0.25, volatility: 4.2 },
    { symbol: 'XAGUSD', name: 'Silver', correlation: 0.78, volatility: 1.8 },
    { symbol: 'EURUSD', name: 'Euro', correlation: 0.45, volatility: 0.7 },
    { symbol: 'GBPUSD', name: 'Pound', correlation: 0.52, volatility: 0.8 }
  ];

  function generateMarketData() {
    const basePrice = selectedAsset === 'XAUUSD' ? 2650 : 
                      selectedAsset === 'BTCUSD' ? 95000 :
                      selectedAsset === 'US500' ? 5900 :
                      selectedAsset === 'ETHUSD' ? 3400 :
                      selectedAsset === 'XAGUSD' ? 31 : 1.08;
    
    const price = basePrice + (Math.random() - 0.5) * basePrice * 0.02;
    const trend = Math.random() > 0.5 ? 'BULLISH' : 'BEARISH';
    const strength = 60 + Math.random() * 35;
    
    const mlScore = calculateMLScore(price, trend, strength);
    const signal = mlScore > 70 ? 'STRONG BUY' : 
                   mlScore > 55 ? 'BUY' :
                   mlScore < 30 ? 'STRONG SELL' :
                   mlScore < 45 ? 'SELL' : 'WAIT';
    
    const volatility = 0.8 + Math.random() * 2.5;
    const sharpeRatio = 1.2 + Math.random() * 1.8;
    const maxDrawdown = -(5 + Math.random() * 15);
    const winRate = 55 + Math.random() * 25;
    
    const patterns = detectPatterns();
    
    const orderFlow = {
      buyVolume: Math.random() * 100,
      sellVolume: Math.random() * 100,
      institutionalFlow: (Math.random() - 0.5) * 50
    };

    return {
      price,
      signal,
      mlScore,
      trend,
      strength,
      entry: price * (signal.includes('BUY') ? 1.001 : 0.999),
      stopLoss: price * (signal.includes('BUY') ? 0.985 : 1.015),
      takeProfit: price * (signal.includes('BUY') ? 1.035 : 0.965),
      confidence: strength,
      volatility,
      sharpeRatio,
      maxDrawdown,
      winRate,
      patterns,
      orderFlow,
      timestamp: new Date().toISOString()
    };
  }

  function calculateMLScore(price, trend, strength) {
    const trendScore = trend === 'BULLISH' ? 30 : 20;
    const momentumScore = strength * 0.4;
    const volumeScore = Math.random() * 20;
    const sentimentScore = Math.random() * 15;
    const macroScore = Math.random() * 10;
    
    return Math.min(100, trendScore + momentumScore + volumeScore + sentimentScore + macroScore);
  }

  function detectPatterns() {
    const allPatterns = [
      { name: 'Head & Shoulders', probability: 0.75, bullish: false },
      { name: 'Double Bottom', probability: 0.82, bullish: true },
      { name: 'Bull Flag', probability: 0.68, bullish: true },
      { name: 'Ascending Triangle', probability: 0.71, bullish: true },
      { name: 'Bear Flag', probability: 0.65, bullish: false },
      { name: 'Cup & Handle', probability: 0.79, bullish: true }
    ];
    
    return allPatterns
      .sort(() => Math.random() - 0.5)
      .slice(0, 2 + Math.floor(Math.random() * 2));
  }

  useEffect(() => {
    const interval = setInterval(() => {
      setLiveData(generateMarketData());
    }, 2000);
    return () => clearInterval(interval);
  }, [selectedAsset, timeframe]);

  const historicalPerformance = useMemo(() => {
    return Array.from({ length: 30 }, (_, i) => ({
      date: `Day ${i + 1}`,
      pnl: (Math.random() - 0.45) * 5000,
      cumulative: i * 200 + Math.random() * 1000,
      drawdown: -(Math.random() * 8)
    }));
  }, [selectedAsset]);

  const correlationMatrix = useMemo(() => {
    return assets.map(asset => ({
      asset: asset.symbol,
      correlation: -1 + Math.random() * 2,
      strength: Math.abs(-1 + Math.random() * 2)
    }));
  }, []);

  const factorAnalysis = useMemo(() => [
    { factor: 'Momentum', value: 75, weight: 0.25 },
    { factor: 'Mean Reversion', value: 45, weight: 0.15 },
    { factor: 'Volatility', value: 68, weight: 0.20 },
    { factor: 'Volume', value: 82, weight: 0.15 },
    { factor: 'Sentiment', value: 58, weight: 0.10 },
    { factor: 'Macro', value: 71, weight: 0.15 }
  ], []);

  const getSignalColor = (signal) => {
    if (signal && signal.includes('BUY')) return '#10b981';
    if (signal && signal.includes('SELL')) return '#ef4444';
    return '#6b7280';
  };

  const getSignalIcon = (signal) => {
    if (signal && signal.includes('BUY')) return <TrendingUp className="w-8 h-8" />;
    if (signal && signal.includes('SELL')) return <TrendingDown className="w-8 h-8" />;
    return <Radio className="w-8 h-8" />;
  };

  if (!liveData) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-16 h-16 animate-pulse mx-auto mb-4 text-emerald-500" />
          <div className="text-2xl font-bold">Initializing TITAN Oracle...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <Shield className="w-12 h-12 text-emerald-500" />
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                TITAN ORACLE
              </h1>
              <p className="text-gray-400 text-sm">Institutional-Grade Trading Intelligence</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className={`px-4 py-2 rounded-lg ${isLive ? 'bg-emerald-900/30 border border-emerald-500' : 'bg-gray-800'}`}>
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-emerald-500 animate-pulse' : 'bg-gray-500'}`} />
                <span className="text-sm font-mono">{isLive ? 'LIVE' : 'PAUSED'}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex gap-2 flex-wrap">
          {assets.map(asset => (
            <button
              key={asset.symbol}
              onClick={() => setSelectedAsset(asset.symbol)}
              className={`px-4 py-2 rounded-lg font-mono text-sm transition-all ${
                selectedAsset === asset.symbol
                  ? 'bg-emerald-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {asset.symbol}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        
        <div className="col-span-12 lg:col-span-4 space-y-6">
          
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-300">AI SIGNAL</h3>
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            
            <div className="text-center mb-6">
              <div className="flex items-center justify-center gap-3 mb-2" style={{ color: getSignalColor(liveData.signal) }}>
                {getSignalIcon(liveData.signal)}
                <div className="text-5xl font-black">{liveData.signal}</div>
              </div>
              <div className="text-gray-400 text-sm mb-4">{selectedAsset} • {timeframe}</div>
              
              <div className="mb-4">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>ML Confidence</span>
                  <span>{liveData.mlScore.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3">
                  <div 
                    className="h-3 rounded-full transition-all duration-500"
                    style={{ 
                      width: `${liveData.mlScore}%`,
                      background: `linear-gradient(90deg, ${liveData.mlScore > 70 ? '#10b981' : liveData.mlScore > 50 ? '#f59e0b' : '#ef4444'}, ${liveData.mlScore > 70 ? '#34d399' : liveData.mlScore > 50 ? '#fbbf24' : '#f87171'})`
                    }}
                  />
                </div>
              </div>

              <div className="text-xs text-gray-500">
                Last Update: {new Date(liveData.timestamp).toLocaleTimeString()}
              </div>
            </div>

            {liveData.signal !== 'WAIT' && (
              <div className="space-y-3">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-gray-400">CURRENT PRICE</span>
                    <Target className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="text-2xl font-bold text-cyan-400 font-mono">
                    ${liveData.price.toFixed(2)}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-cyan-900/20 border border-cyan-700 rounded-lg p-3">
                    <div className="text-xs text-cyan-400 mb-1">ENTRY</div>
                    <div className="text-sm font-bold font-mono">${liveData.entry.toFixed(2)}</div>
                  </div>
                  <div className="bg-red-900/20 border border-red-700 rounded-lg p-3">
                    <div className="text-xs text-red-400 mb-1">STOP</div>
                    <div className="text-sm font-bold font-mono">${liveData.stopLoss.toFixed(2)}</div>
                  </div>
                  <div className="bg-emerald-900/20 border border-emerald-700 rounded-lg p-3">
                    <div className="text-xs text-emerald-400 mb-1">TARGET</div>
                    <div className="text-sm font-bold font-mono">${liveData.takeProfit.toFixed(2)}</div>
                  </div>
                </div>

                <div className="bg-gray-800 rounded-lg p-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Risk/Reward Ratio</span>
                    <span className="text-emerald-400 font-bold">
                      1:{(Math.abs(liveData.takeProfit - liveData.entry) / Math.abs(liveData.entry - liveData.stopLoss)).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-300">PATTERNS DETECTED</h3>
              <Zap className="w-5 h-5 text-yellow-400" />
            </div>
            <div className="space-y-3">
              {liveData.patterns.map((pattern, idx) => (
                <div key={idx} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-semibold">{pattern.name}</span>
                    <span className={`text-xs px-2 py-1 rounded ${pattern.bullish ? 'bg-emerald-900/30 text-emerald-400' : 'bg-red-900/30 text-red-400'}`}>
                      {pattern.bullish ? 'BULL' : 'BEAR'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${pattern.bullish ? 'bg-emerald-500' : 'bg-red-500'}`}
                        style={{ width: `${pattern.probability * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400">{(pattern.probability * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-300">RISK METRICS</h3>
              <Shield className="w-5 h-5 text-purple-400" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Volatility</div>
                <div className="text-lg font-bold text-orange-400">{liveData.volatility.toFixed(2)}%</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Sharpe Ratio</div>
                <div className="text-lg font-bold text-cyan-400">{liveData.sharpeRatio.toFixed(2)}</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Max Drawdown</div>
                <div className="text-lg font-bold text-red-400">{liveData.maxDrawdown.toFixed(1)}%</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Win Rate</div>
                <div className="text-lg font-bold text-emerald-400">{liveData.winRate.toFixed(0)}%</div>
              </div>
            </div>
          </div>

        </div>

        <div className="col-span-12 lg:col-span-5 space-y-6">
          
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-300 mb-4">CUMULATIVE P&L</h3>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={historicalPerformance}>
                <defs>
                  <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#9ca3af' }}
                />
                <Area type="monotone" dataKey="cumulative" stroke="#10b981" fillOpacity={1} fill="url(#pnlGradient)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-300 mb-4">MULTI-FACTOR ANALYSIS</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={factorAnalysis}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="factor" stroke="#9ca3af" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#6b7280" />
                <Radar name="Score" dataKey="value" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
              </RadarChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-3 gap-2">
              {factorAnalysis.map(factor => (
                <div key={factor.factor} className="text-center">
                  <div className="text-xs text-gray-400">{factor.factor}</div>
                  <div className="text-sm font-bold text-purple-400">{factor.value}</div>
                  <div className="text-xs text-gray-500">W: {(factor.weight * 100).toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-300 mb-4">ORDER FLOW ANALYSIS</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[liveData.orderFlow]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
                <Bar dataKey="buyVolume" fill="#10b981" name="Buy Volume" />
                <Bar dataKey="sellVolume" fill="#ef4444" name="Sell Volume" />
                <Bar dataKey="institutionalFlow" fill="#8b5cf6" name="Institutional" />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="bg-emerald-900/20 border border-emerald-700 rounded-lg p-3 text-center">
                <div className="text-xs text-emerald-400 mb-1">Net Buy Pressure</div>
                <div className="text-xl font-bold text-emerald-400">
                  {((liveData.orderFlow.buyVolume / (liveData.orderFlow.buyVolume + liveData.orderFlow.sellVolume)) * 100).toFixed(0)}%
                </div>
              </div>
              <div className="bg-purple-900/20 border border-purple-700 rounded-lg p-3 text-center">
                <div className="text-xs text-purple-400 mb-1">Smart Money</div>
                <div className="text-xl font-bold text-purple-400">
                  {liveData.orderFlow.institutionalFlow > 0 ? '+' : ''}{liveData.orderFlow.institutionalFlow.toFixed(1)}M
                </div>
              </div>
            </div>
          </div>

        </div>

        <div className="col-span-12 lg:col-span-3 space-y-6">
          
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-300">MARKET REGIME</h3>
              <Activity className="w-5 h-5 text-blue-400" />
            </div>
            <div className="space-y-3">
              {['TRENDING', 'RANGING', 'VOLATILE', 'CALM'].map(regime => (
                <div 
                  key={regime}
                  className={`p-3 rounded-lg border transition-all ${
                    marketRegime === regime 
                      ? 'bg-blue-900/30 border-blue-500' 
                      : 'bg-gray-800 border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold">{regime}</span>
                    {marketRegime === regime && (
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-300 mb-4">CROSS-ASSET CORRELATION</h3>
            <div className="space-y-2">
              {correlationMatrix.slice(0, 5).map((item, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <span className="text-xs font-mono text-gray-400 w-16">{item.asset}</span>
                  <div className="flex-1 bg-gray-800 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${item.correlation > 0 ? 'bg-emerald-500' : 'bg-red-500'}`}
                      style={{ width: `${Math.abs(item.correlation) * 50}%`, marginLeft: item.correlation < 0 ? `${50 - Math.abs(item.correlation) * 50}%` : '50%' }}
                    />
                  </div>
                  <span className={`text-xs font-bold w-12 text-right ${item.correlation > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {item.correlation.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-300 mb-4">HIGH IMPACT EVENTS</h3>
            <div className="space-y-3">
              {[
                { event: 'Fed Rate Decision', time: '14:00 EST', impact: 'HIGH' },
                { event: 'NFP Report', time: '08:30 EST', impact: 'HIGH' },
                { event: 'CPI Data', time: '08:30 EST', impact: 'MEDIUM' }
              ].map((item, idx) => (
                <div key={idx} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex items-start justify-between mb-1">
                    <span className="text-sm font-semibold text-gray-200">{item.event}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      item.impact === 'HIGH' ? 'bg-red-900/30 text-red-400' : 'bg-yellow-900/30 text-yellow-400'
                    }`}>
                      {item.impact}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">{item.time}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h3 className="text-lg font-bold text-gray-300 mb-4">SYSTEM STATUS</h3>
            <div className="space-y-2">
              {[
                { name: 'Data Feed', status: 'ACTIVE', color: 'emerald' },
                { name: 'ML Engine', status: 'ACTIVE', color: 'emerald' },
                { name: 'Risk Manager', status: 'ACTIVE', color: 'emerald' },
                { name: 'Order Router', status: 'STANDBY', color: 'yellow' }
              ].map((sys, idx) => (
                <div key={idx} className="flex items-center justify-between bg-gray-800 rounded-lg p-2">
                  <span className="text-xs text-gray-400">{sys.name}</span>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${sys.color === 'emerald' ? 'bg-emerald-500' : 'bg-yellow-500'} ${sys.status === 'ACTIVE' ? 'animate-pulse' : ''}`} />
                    <span className={`text-xs font-mono ${sys.color === 'emerald' ? 'text-emerald-400' : 'text-yellow-400'}`}>{sys.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>

      <div className="mt-6 text-center text-xs text-gray-600">
        <p>TITAN Oracle v4.0 | Institutional-Grade Trading Intelligence</p>
      </div>
    </div>
  );
};

export default TitanOracleTerminal;
