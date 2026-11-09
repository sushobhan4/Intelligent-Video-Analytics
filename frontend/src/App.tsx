import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ShieldCheck,
  Play,
  Square,
  Search,
  Trash2,
  Clock,
  Compass,
  UploadCloud,
  Cpu,
  Film,
  X,
  RefreshCw,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  FileVideo,
  ArrowUpDown,
  Eye,
  Calendar,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Activity,
  HardDrive,
  Database,
  Zap,
  Check,
  SlidersHorizontal,
  Info
} from 'lucide-react';
import './App.css';

// -----------------------------------------------------------------------------
// Data Types
// -----------------------------------------------------------------------------
export type TabKey = 'ingest' | 'clips' | 'system';

export type SortOption =
  | 'processed'
  | 'unprocessed'
  | 'ascending'
  | 'descending'
  | 'length'
  | 'size';

interface SystemStatus {
  status: string;
  gpu: {
    available: boolean;
    name: string;
    vram_total_gb: number;
    vram_allocated_gb: number;
    vram_free_gb: number;
    vram_percent: number;
    device_capability: string;
  };
  cpu: {
    name: string;
    cores: number;
    threads: number;
    usage_pct: number;
    architecture: string;
  };
  ram: {
    total_gb: number;
    used_gb: number;
    free_gb: number;
    percent: number;
  };
  vlm_model: {
    name: string;
    quantization: string;
    attention: string;
    device: string;
    context_tracking?: string;
  };
  embedding_model: {
    name: string;
    dimensions: number;
    context_window: number;
    device: string;
    asymmetric: boolean;
  };
  vector_db: {
    name: string;
    collection: string;
    distance_metric: string;
    total_indexed_clips: number;
    total_disk_clips: number;
    unprocessed_clips: number;
    storage_path: string;
  };
}

interface VideoFile {
  name: string;
  path: string;
  category: string;
  size_mb: number;
  url: string;
}

interface ClipEvent {
  clip_id: string;
  status: 'processed' | 'unprocessed';
  similarity?: number | null;
  confidence_pct?: number | null;
  title: string;
  caption: string;
  timestamp: string;
  action_summary: string;
  camera_id: string;
  start_sec: number;
  duration_sec: number;
  size_mb: number;
  date: string;
  time_range: string;
  video_url: string;
  thumbnail_url: string;
  full_report: string;
}

interface PipelineState {
  status: 'idle' | 'clipping' | 'analyzing' | 'completed' | 'error' | 'stopped';
  video_path: string;
  video_name: string;
  stage: string;
  progress_pct: number;
  scan_fps: number;
  total_clips: number;
  processed_clips: number;
  active_clip?: string | null;
  elapsed_seconds: number;
  eta_seconds: number;
  error_message?: string | null;
}

const SORT_LABELS: Record<SortOption, { label: string; desc: string }> = {
  ascending: { label: 'Ascending (Timeline)', desc: 'Earliest events first (00:00 ➔ 23:59)' },
  descending: { label: 'Descending (Timeline)', desc: 'Latest events first (23:59 ➔ 00:00)' },
  processed: { label: 'Processed First', desc: 'Clips analyzed by VLM & indexed' },
  unprocessed: { label: 'Unprocessed First', desc: 'Clips pending VLM analysis' },
  length: { label: 'Length (Duration)', desc: 'Longest event duration first' },
  size: { label: 'Size (MB)', desc: 'Largest file size first' }
};

export function App() {
  // Navigation Tabs State (Apple Segmented Control)
  const [activeTab, setActiveTab] = useState<TabKey>('clips');

  // System Diagnostics State
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean>(false);
  const [realtimeConnected, setRealtimeConnected] = useState<boolean>(false);

  // Video Library & Ingestion State (Tab 1)
  const [videos, setVideos] = useState<VideoFile[]>([]);
  const [selectedVideoPath, setSelectedVideoPath] = useState<string>('');
  const [uploading, setUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Autonomous Pipeline Execution State
  const [pipelineState, setPipelineState] = useState<PipelineState>({
    status: 'idle',
    video_path: '',
    video_name: '',
    stage: 'Idle',
    progress_pct: 0,
    scan_fps: 0,
    total_clips: 0,
    processed_clips: 0,
    elapsed_seconds: 0,
    eta_seconds: 0
  });

  // Clips Explorer State (Tab 2)
  const [clips, setClips] = useState<ClipEvent[]>([]);
  const [loadingClips, setLoadingClips] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [activeSearch, setActiveSearch] = useState<string>('');
  const [sortBy, setSortBy] = useState<SortOption>('ascending');
  const [selectedDate, setSelectedDate] = useState<string>('');
  const [availableDates, setAvailableDates] = useState<string[]>([]);
  const [totalProcessed, setTotalProcessed] = useState<number>(0);
  const [totalUnprocessed, setTotalUnprocessed] = useState<number>(0);

  // Persistent refs for background silent sync without reconnecting sockets
  const activeSearchRef = useRef(activeSearch);
  activeSearchRef.current = activeSearch;
  const sortByRef = useRef(sortBy);
  sortByRef.current = sortBy;
  const selectedDateRef = useRef(selectedDate);
  selectedDateRef.current = selectedDate;

  // Popover menus state
  const [sortMenuOpen, setSortMenuOpen] = useState<boolean>(false);
  const [calendarMenuOpen, setCalendarMenuOpen] = useState<boolean>(false);
  const sortMenuRef = useRef<HTMLDivElement>(null);
  const calendarMenuRef = useRef<HTMLDivElement>(null);

  // On-demand single clip analysis
  const [analyzingClipId, setAnalyzingClipId] = useState<string | null>(null);

  // Interactive Video & Dossier Modal
  const [activeModalClip, setActiveModalClip] = useState<ClipEvent | null>(null);

  // Feedback Notification Toast
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // Close popover menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sortMenuRef.current && !sortMenuRef.current.contains(e.target as Node)) {
        setSortMenuOpen(false);
      }
      if (calendarMenuRef.current && !calendarMenuRef.current.contains(e.target as Node)) {
        setCalendarMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // 1. Fetch System & Hardware Diagnostics
  const fetchSystemStatus = async () => {
    try {
      const res = await fetch('/api/system/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
        setServerOnline(true);
      } else {
        setServerOnline(false);
      }
    } catch {
      setServerOnline(false);
    }
  };

  // 2. Fetch Available Videos
  const fetchVideos = async () => {
    try {
      const res = await fetch('/api/videos');
      if (res.ok) {
        const data = await res.json();
        setVideos(data.videos || []);
        if (data.videos && data.videos.length > 0 && !selectedVideoPath) {
          setSelectedVideoPath(data.videos[0].path);
        }
      }
    } catch (err) {
      console.error('Failed to fetch videos:', err);
    }
  };

  // 3. Fetch Clips (1-Day View with Sort & Search)
  const fetchClips = useCallback(
    async (
      query: string = '',
      sort: SortOption = sortBy,
      date: string = selectedDate,
      silent: boolean = false
    ) => {
      if (!silent) {
        setLoadingClips(true);
      }
      try {
        const params = new URLSearchParams();
        if (query.trim()) {
          params.append('search', query.trim());
        }
        params.append('sort_by', sort);
        if (date) {
          params.append('date', date);
        }

        const res = await fetch(`/api/clips?${params.toString()}`);
        if (res.ok) {
          const data = await res.json();
          setClips(data.clips || []);
          setTotalProcessed(data.total_processed || 0);
          setTotalUnprocessed(data.total_unprocessed || 0);
          if (data.available_dates && data.available_dates.length > 0) {
            setAvailableDates(data.available_dates);
            if (!date || !data.available_dates.includes(date)) {
              setSelectedDate(data.selected_date || data.available_dates[0]);
            }
          }
          setActiveSearch(query.trim());
        }
      } catch (err) {
        console.error('Error fetching clips:', err);
      } finally {
        if (!silent) {
          setLoadingClips(false);
        }
      }
    },
    [sortBy, selectedDate]
  );

  // Stable silent refresh function for background realtime events
  const refreshClipsSilent = useCallback(() => {
    fetchClips(activeSearchRef.current, sortByRef.current, selectedDateRef.current, true);
  }, [fetchClips]);

  // Initial Data Fetch
  useEffect(() => {
    fetchSystemStatus();
    fetchVideos();
    fetchClips('', sortBy, selectedDate);
  }, []);

  // 4. Real-time WebSocket + SSE Auto-Sync Manager
  useEffect(() => {
    let ws: WebSocket | null = null;
    let eventSource: EventSource | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let isMounted = true;

    const handlePayload = (payload: any) => {
      if (!isMounted) return;

      if (payload.type === 'pipeline_state' && payload.data) {
        setPipelineState(payload.data);
        // Refresh clips silently in place as each clip is analyzed
        refreshClipsSilent();
        if (payload.data.status === 'completed') {
          fetchSystemStatus();
        }
      } else if (payload.type === 'clips_updated' || payload.type === 'clip_updated') {
        refreshClipsSilent();
        fetchSystemStatus();
      } else if (payload.type === 'videos_updated') {
        fetchVideos();
      } else if (payload.type === 'system_telemetry' && payload.data) {
        setSystemStatus(payload.data);
        setServerOnline(true);
      }
    };

    const connectSSE = () => {
      if (eventSource || !isMounted) return;
      try {
        eventSource = new EventSource('/api/events');
        eventSource.onopen = () => {
          if (isMounted) {
            setRealtimeConnected(true);
            setServerOnline(true);
          }
        };
        eventSource.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            handlePayload(payload);
          } catch {}
        };
        eventSource.onerror = () => {
          eventSource?.close();
          eventSource = null;
        };
      } catch {}
    };

    const connectWebSocket = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/pipeline`;

      try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          if (isMounted) {
            setRealtimeConnected(true);
            setServerOnline(true);
          }
        };

        ws.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            handlePayload(payload);
          } catch {}
        };

        ws.onclose = () => {
          if (isMounted) {
            setRealtimeConnected(false);
            connectSSE();
            reconnectTimeout = setTimeout(connectWebSocket, 2500);
          }
        };

        ws.onerror = () => {
          ws?.close();
        };
      } catch {
        connectSSE();
      }
    };

    connectWebSocket();

    // Fallback heartbeat telemetry polling every 4 seconds in case network dropped
    const heartbeatTimer = setInterval(() => {
      fetchSystemStatus();
    }, 4000);

    return () => {
      isMounted = false;
      if (ws) ws.close();
      if (eventSource) eventSource.close();
      clearTimeout(reconnectTimeout);
      clearInterval(heartbeatTimer);
    };
  }, [refreshClipsSilent]);

  // Video Upload
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!serverOnline) {
      alert('Backend server is offline! Start it by running: python server.py');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const uploaded = await res.json();
        await fetchVideos();
        setSelectedVideoPath(uploaded.path);
        showToast(`Uploaded "${uploaded.filename}" (${uploaded.size_mb} MB)`);
      } else {
        const errData = await res.json().catch(() => ({}));
        alert(`Upload failed: ${errData.detail || 'Server error'}`);
      }
    } catch (err: any) {
      alert(`Upload error: ${err.message || err}`);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Start Autonomous Pipeline
  const handleStartPipeline = async () => {
    if (!selectedVideoPath) {
      alert('Please select or upload a video first.');
      return;
    }

    try {
      const res = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: selectedVideoPath,
          camera_id: 'FRONT_PORCH_CAM_01',
          chunk_sec: 10.0,
          resolution: '640x360',
          pixel_threshold: 18,
          min_motion_percent: 0.5
        })
      });

      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || 'Could not start pipeline.');
      } else {
        showToast('Autonomous pipeline launched in background.');
      }
    } catch (err) {
      alert(`Error starting pipeline: ${err}`);
    }
  };

  // Stop Pipeline
  const handleStopPipeline = async () => {
    try {
      await fetch('/api/pipeline/stop', { method: 'POST' });
      showToast('Pipeline cancellation requested.');
    } catch (err) {
      console.error('Stop request failed:', err);
    }
  };

  // Process All Unprocessed Clips Batch
  const handleProcessAllUnprocessed = async () => {
    const pendingCount = systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed;
    if (pendingCount === 0) {
      alert('All clips on disk are already processed and indexed in ChromaDB.');
      return;
    }

    try {
      const res = await fetch('/api/pipeline/process-unprocessed', {
        method: 'POST'
      });
      if (res.ok) {
        showToast(`Batch processing started for ${pendingCount} unprocessed clips.`);
      } else {
        const err = await res.json().catch(() => ({}));
        alert(err.detail || 'Could not start batch processing.');
      }
    } catch (err: any) {
      alert(`Batch process error: ${err.message || err}`);
    }
  };

  // Analyze Single Unprocessed Clip On-Demand
  const handleAnalyzeSingleClip = async (clipId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (analyzingClipId) return;

    setAnalyzingClipId(clipId);
    showToast(`Analyzing clip '${clipId}' with Qwen3-VL...`);

    try {
      const res = await fetch(`/api/clips/${clipId}/analyze`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        showToast(`Analyzed & Indexed: "${data.title}"`);
        await fetchClips(activeSearch, sortBy, selectedDate);
        await fetchSystemStatus();
      } else {
        const err = await res.json().catch(() => ({}));
        alert(`Analysis failed: ${err.detail || 'Unknown error'}`);
      }
    } catch (err: any) {
      alert(`Error analyzing clip: ${err.message || err}`);
    } finally {
      setAnalyzingClipId(null);
    }
  };

  // Semantic Search Submit
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    fetchClips(searchQuery, sortBy, selectedDate);
  };

  // Clear Search
  const handleClearSearch = () => {
    setSearchQuery('');
    setActiveSearch('');
    fetchClips('', sortBy, selectedDate);
  };

  // Handle Sort Change
  const handleSelectSort = (newSort: SortOption) => {
    setSortBy(newSort);
    setSortMenuOpen(false);
    fetchClips(activeSearch, newSort, selectedDate);
  };

  // Handle Date Navigation
  const handleSelectDate = (newDate: string) => {
    setSelectedDate(newDate);
    setCalendarMenuOpen(false);
    fetchClips(activeSearch, sortBy, newDate);
  };

  const handleDateStep = (direction: 'prev' | 'next') => {
    if (availableDates.length <= 1) return;
    const currentIndex = availableDates.indexOf(selectedDate);
    if (currentIndex === -1) return;

    let targetIndex = currentIndex;
    if (direction === 'next' && currentIndex > 0) {
      // dates are reverse sorted (newest at index 0)
      targetIndex = currentIndex - 1;
    } else if (direction === 'prev' && currentIndex < availableDates.length - 1) {
      targetIndex = currentIndex + 1;
    }

    if (targetIndex !== currentIndex) {
      handleSelectDate(availableDates[targetIndex]);
    }
  };

  // Reset Vector DB
  const handleResetDB = async () => {
    if (!confirm('Are you sure you want to completely erase all indexed vectors from ChromaDB? Motion clips on disk will remain intact as unprocessed.')) return;
    try {
      const res = await fetch('/api/clips/reset', { method: 'POST' });
      if (res.ok) {
        fetchClips('', sortBy, selectedDate);
        fetchSystemStatus();
        showToast('ChromaDB vector store cleared.');
      }
    } catch (err) {
      alert(`Reset failed: ${err}`);
    }
  };

  // Delete Individual Clip
  const handleDeleteClip = async (clipId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`Delete clip '${clipId}' from database?`)) return;
    try {
      const res = await fetch(`/api/clips/${clipId}`, { method: 'DELETE' });
      if (res.ok) {
        fetchClips(activeSearch, sortBy, selectedDate);
        fetchSystemStatus();
        showToast(`Clip deleted from ChromaDB.`);
      }
    } catch (err) {
      alert(`Delete error: ${err}`);
    }
  };

  const isPipelineActive =
    pipelineState.status === 'clipping' || pipelineState.status === 'analyzing';

  // Format Date for Display (e.g. 2026-11-04 -> Nov 4, 2026)
  const formatDateDisplay = (dateStr: string) => {
    if (!dateStr) return 'Select Date';
    try {
      const [year, month, day] = dateStr.split('-');
      const d = new Date(parseInt(year), parseInt(month) - 1, parseInt(day));
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return dateStr;
    }
  };

  // Format Card Timestamp for Display (e.g. "28-10-2026 Wed 08:42:03" -> "08:42:03")
  const getDisplayTimestamp = (c: ClipEvent) => {
    if (c.timestamp) {
      const m = c.timestamp.match(/(\d{1,2}:\d{2}(?::\d{2})?)/);
      if (m) {
        return m[1];
      }
      return c.timestamp;
    }
    return c.time_range || '00:00';
  };

  return (
    <div className="apple-app-root">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="apple-toast-banner">
          <Zap size={14} color="#0a84ff" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* ================================================================== */}
      {/* Floating Apple Navigation Bar & Segmented Control                  */}
      {/* ================================================================== */}
      <header className="apple-header-bar">
        <div className="header-inner">
          {/* Brand & System Status */}
          <div className="brand-group">
            <div className="brand-glyph">
              <ShieldCheck size={20} color="#0a84ff" />
            </div>
            <div className="brand-copy">
              <div className="brand-heading">
                Neural CCTV <span className="version-pill">macOS HIG</span>
              </div>
              <div className="brand-subtext">Autonomous Vision Forensic Intelligence</div>
            </div>
          </div>

          {/* Apple Segmented Control Pill (3 Dedicated Tabs) */}
          <nav className="segmented-control" role="tablist">
            <button
              role="tab"
              aria-selected={activeTab === 'ingest'}
              className={`segmented-tab ${activeTab === 'ingest' ? 'active' : ''}`}
              onClick={() => setActiveTab('ingest')}
            >
              <UploadCloud size={15} />
              <span>Studio & Ingest</span>
              {isPipelineActive && <span className="tab-pulsing-dot" />}
            </button>

            <button
              role="tab"
              aria-selected={activeTab === 'clips'}
              className={`segmented-tab ${activeTab === 'clips' ? 'active' : ''}`}
              onClick={() => setActiveTab('clips')}
            >
              <Film size={15} />
              <span>Forensic Clips</span>
              <span className="tab-counter-badge">{clips.length}</span>
            </button>

            <button
              role="tab"
              aria-selected={activeTab === 'system'}
              className={`segmented-tab ${activeTab === 'system' ? 'active' : ''}`}
              onClick={() => setActiveTab('system')}
            >
              <Activity size={15} />
              <span>System & DB</span>
            </button>
          </nav>

          {/* Telemetry Indicator */}
          <div className="header-actions">
            <div className="status-pill">
              <span className={`status-indicator ${realtimeConnected ? 'online pulsing' : serverOnline ? 'online' : 'offline'}`} />
              <span className="status-label">
                {realtimeConnected ? 'Live Realtime Sync' : serverOnline ? 'FastAPI Online' : 'Connecting...'}
              </span>
            </div>

            <button
              className="icon-glass-btn"
              title="Refresh System State"
              onClick={() => {
                fetchSystemStatus();
                fetchClips(activeSearch, sortBy, selectedDate);
              }}
            >
              <RefreshCw size={14} className={loadingClips ? 'spin-animation' : ''} />
            </button>
          </div>
        </div>
      </header>

      {/* Main Tab Content Container */}
      <main className="content-container">
        {/* ================================================================ */}
        {/* TAB 1: VIDEO STUDIO & AUTONOMOUS INGESTION                       */}
        {/* ================================================================ */}
        {activeTab === 'ingest' && (
          <section className="tab-pane fade-in">
            <div className="section-hero">
              <h2 className="section-title">Video Ingestion & Autonomous Processing</h2>
              <p className="section-description">
                Upload raw CCTV recordings or select surveillance feeds. Motion events will be
                automatically isolated via hardware NVDEC frame scanning, analyzed with
                Qwen3-VL, and indexed as 768-d vectors into ChromaDB.
              </p>
            </div>

            <div className="studio-grid">
              {/* Left Column: Input Source & File Upload */}
              <div className="apple-card">
                <div className="card-header">
                  <div className="card-title">
                    <FileVideo size={16} color="#0a84ff" />
                    <span>Video Source Selection</span>
                  </div>
                  <span className="badge-subtle">{videos.length} Available</span>
                </div>

                <div className="card-body">
                  <div className="field-group">
                    <label className="field-label">Select Surveillance Footage</label>
                    <select
                      className="apple-select"
                      value={selectedVideoPath}
                      onChange={(e) => setSelectedVideoPath(e.target.value)}
                      disabled={isPipelineActive}
                    >
                      {videos.map((v) => (
                        <option key={v.path} value={v.path}>
                          {v.name} ({v.size_mb} MB) — [{v.category}]
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Dropzone Upload */}
                  <div className="upload-dropzone" onClick={() => fileInputRef.current?.click()}>
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: 'none' }}
                      accept="video/mp4,video/x-matroska,video/avi,video/quicktime"
                      onChange={handleUpload}
                    />
                    <div className="dropzone-icon">
                      <UploadCloud size={28} color="#0a84ff" />
                    </div>
                    <div className="dropzone-text">
                      <strong>Click to upload new CCTV footage</strong>
                      <span>Supports MP4, MKV, AVI, MOV up to 2GB</span>
                    </div>
                    {uploading && (
                      <div className="upload-status">
                        <RefreshCw size={14} className="spin-animation" />
                        <span>Streaming file to server...</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Right Column: Execution Controller & Settings */}
              <div className="apple-card">
                <div className="card-header">
                  <div className="card-title">
                    <SlidersHorizontal size={16} color="#30d158" />
                    <span>Extraction Parameters</span>
                  </div>
                  <span className="badge-subtle">NVIDIA NVDEC Enabled</span>
                </div>

                <div className="card-body">
                  <div className="param-rows">
                    <div className="param-item">
                      <span className="param-label">Camera Identifier</span>
                      <span className="param-value font-mono">FRONT_PORCH_CAM_01</span>
                    </div>
                    <div className="param-item">
                      <span className="param-label">Motion Chunk Duration</span>
                      <span className="param-value font-mono">10.0 seconds</span>
                    </div>
                    <div className="param-item">
                      <span className="param-label">Scanning Resolution</span>
                      <span className="param-value font-mono">640 × 360 (Fast NVDEC)</span>
                    </div>
                    <div className="param-item">
                      <span className="param-label">Frame Difference Threshold</span>
                      <span className="param-value font-mono">18 / 255</span>
                    </div>
                    <div className="param-item">
                      <span className="param-label">Min Motion Cluster</span>
                      <span className="param-value font-mono">0.50%</span>
                    </div>
                  </div>

                  <div className="button-action-row studio-action-col">
                    {!isPipelineActive ? (
                      <>
                        <button
                          className="apple-primary-btn"
                          onClick={handleStartPipeline}
                          disabled={!selectedVideoPath || !serverOnline}
                        >
                          <Play size={16} fill="#fff" />
                          <span>Start Autonomous Pipeline</span>
                        </button>

                        <button
                          className="apple-amber-btn"
                          onClick={handleProcessAllUnprocessed}
                          disabled={
                            !serverOnline ||
                            (systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed) === 0
                          }
                          title={
                            (systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed) > 0
                              ? `Run Qwen3-VL analysis and ChromaDB indexing on all ${
                                  systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed
                                } pending clips`
                              : 'No unprocessed clips pending'
                          }
                        >
                          <Zap size={15} color="#ff9f0a" />
                          <span>
                            Process All Unprocessed Clips
                            {(systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed) > 0
                              ? ` (${systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed} Pending)`
                              : ' (0 Pending)'}
                          </span>
                        </button>
                      </>
                    ) : (
                      <button className="apple-danger-btn" onClick={handleStopPipeline}>
                        <Square size={15} fill="#fff" />
                        <span>Stop Pipeline</span>
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Unprocessed Clips Notice Capsule */}
            {(systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed) > 0 && !isPipelineActive && (
              <div className="unprocessed-notice-capsule">
                <div className="capsule-left">
                  <Zap size={16} color="#ff9f0a" />
                  <span>
                    <strong>{systemStatus?.vector_db.unprocessed_clips ?? totalUnprocessed} motion events</strong> on disk are awaiting Qwen3-VL multimodal forensic analysis.
                  </span>
                </div>
                <button
                  className="capsule-action-btn"
                  onClick={handleProcessAllUnprocessed}
                >
                  ⚡ Start Batch Analysis Now
                </button>
              </div>
            )}

            {/* Live Progress Bar Card during processing */}
            {(isPipelineActive ||
              pipelineState.status === 'completed' ||
              pipelineState.status === 'error') && (
              <div className="apple-card live-pipeline-card">
                <div className="card-header">
                  <div className="card-title">
                    {isPipelineActive && (
                      <RefreshCw size={16} className="spin-animation" color="#0a84ff" />
                    )}
                    {pipelineState.status === 'completed' && (
                      <CheckCircle2 size={16} color="#30d158" />
                    )}
                    {pipelineState.status === 'error' && (
                      <AlertCircle size={16} color="#ff453a" />
                    )}
                    <span>
                      {isPipelineActive ? 'Autonomous Pipeline in Progress' : 'Pipeline State'}
                    </span>
                  </div>
                  <span className="stage-tag">{pipelineState.stage}</span>
                </div>

                <div className="card-body">
                  <div className="progress-track-wrapper">
                    <div className="progress-info-bar">
                      <span className="progress-pct-label">{pipelineState.progress_pct}%</span>
                      <span className="progress-video-name">
                        {pipelineState.video_name || 'Surveillance Feed'}
                      </span>
                    </div>
                    <div className="apple-progress-bar">
                      <div
                        className="apple-progress-fill"
                        style={{ width: `${Math.max(4, pipelineState.progress_pct)}%` }}
                      />
                    </div>
                  </div>

                  <div className="telemetry-grid">
                    <div className="telemetry-cell">
                      <span className="cell-label">Scanning Speed</span>
                      <span className="cell-value">
                        {pipelineState.scan_fps > 0 ? `${pipelineState.scan_fps} FPS` : 'Ready'}
                      </span>
                    </div>

                    <div className="telemetry-cell">
                      <span className="cell-label">Clips Extracted</span>
                      <span className="cell-value">
                        {pipelineState.processed_clips} / {pipelineState.total_clips}
                      </span>
                    </div>

                    <div className="telemetry-cell">
                      <span className="cell-label">Elapsed Time</span>
                      <span className="cell-value">{pipelineState.elapsed_seconds}s</span>
                    </div>

                    <div className="telemetry-cell">
                      <span className="cell-label">Estimated Remaining</span>
                      <span className="cell-value">
                        {pipelineState.eta_seconds > 0
                          ? `~${Math.round(pipelineState.eta_seconds)}s`
                          : '—'}
                      </span>
                    </div>
                  </div>

                  {pipelineState.status === 'completed' && (
                    <div className="pipeline-complete-callout">
                      <div className="callout-text">
                        <CheckCircle2 size={18} color="#30d158" />
                        <span>
                          Motion analysis complete! All events extracted, tagged, and indexed.
                        </span>
                      </div>
                      <button
                        className="apple-secondary-btn"
                        onClick={() => setActiveTab('clips')}
                      >
                        <span>Explore Clips</span>
                        <ChevronRight size={14} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>
        )}

        {/* ================================================================ */}
        {/* TAB 2: FORENSIC CLIPS EXPLORER (1-DAY VIEW)                      */}
        {/* ================================================================ */}
        {activeTab === 'clips' && (
          <section className="tab-pane fade-in">
            {/* Top Toolbar: Search + Sort + Calendar */}
            <div className="clips-toolbar-container">
              <div className="toolbar-main-row">
                {/* 1. Search Bar */}
                <form className="search-field-wrapper" onSubmit={handleSearchSubmit}>
                  <Search size={16} className="search-field-icon" />
                  <input
                    type="text"
                    className="apple-search-input"
                    placeholder="Search CCTV events... (e.g., 'a brown dog at the gate', 'woman in orange sari', 'blue auto-rickshaw')"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  {searchQuery && (
                    <button
                      type="button"
                      className="search-field-clear"
                      onClick={handleClearSearch}
                      title="Clear search query"
                    >
                      <X size={14} />
                    </button>
                  )}
                  <button type="submit" className="search-submit-btn">
                    <Sparkles size={14} />
                    <span>Search</span>
                  </button>
                </form>

                {/* 2. Sorting Popover Button */}
                <div className="popover-anchor" ref={sortMenuRef}>
                  <button
                    className={`apple-toolbar-btn ${sortMenuOpen ? 'active' : ''}`}
                    onClick={() => {
                      setSortMenuOpen(!sortMenuOpen);
                      setCalendarMenuOpen(false);
                    }}
                    title="Sort motion clips"
                  >
                    <ArrowUpDown size={14} />
                    <span className="btn-text">
                      Sort: <strong>{SORT_LABELS[sortBy]?.label.split(' ')[0] || sortBy}</strong>
                    </span>
                    <ChevronDown size={13} className={`chevron ${sortMenuOpen ? 'up' : ''}`} />
                  </button>

                  {sortMenuOpen && (
                    <div className="apple-menu-popover sort-popover fade-in-scale">
                      <div className="menu-header">Sort Clips By</div>
                      {(
                        [
                          'ascending',
                          'descending',
                          'processed',
                          'unprocessed',
                          'length',
                          'size'
                        ] as SortOption[]
                      ).map((option) => {
                        const isSelected = sortBy === option;
                        return (
                          <button
                            key={option}
                            className={`menu-item ${isSelected ? 'selected' : ''}`}
                            onClick={() => handleSelectSort(option)}
                          >
                            <div className="menu-item-text">
                              <span className="item-title">{SORT_LABELS[option].label}</span>
                              <span className="item-desc">{SORT_LABELS[option].desc}</span>
                            </div>
                            {isSelected && <Check size={14} className="item-check" />}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* 3. Calendar (1-Day View) Navigation Button */}
                <div className="popover-anchor" ref={calendarMenuRef}>
                  <div className="date-nav-cluster">
                    <button
                      className="nav-arrow-btn"
                      onClick={() => handleDateStep('prev')}
                      title="Previous recorded day"
                      disabled={availableDates.indexOf(selectedDate) >= availableDates.length - 1}
                    >
                      <ChevronLeft size={14} />
                    </button>

                    <button
                      className={`apple-toolbar-btn date-pill-btn ${calendarMenuOpen ? 'active' : ''}`}
                      onClick={() => {
                        setCalendarMenuOpen(!calendarMenuOpen);
                        setSortMenuOpen(false);
                      }}
                      title="Select day of clips"
                    >
                      <Calendar size={14} />
                      <span className="btn-text">
                        <strong>{formatDateDisplay(selectedDate)}</strong>
                      </span>
                      <ChevronDown size={13} className={`chevron ${calendarMenuOpen ? 'up' : ''}`} />
                    </button>

                    <button
                      className="nav-arrow-btn"
                      onClick={() => handleDateStep('next')}
                      title="Next recorded day"
                      disabled={availableDates.indexOf(selectedDate) <= 0}
                    >
                      <ChevronRight size={14} />
                    </button>
                  </div>

                  {calendarMenuOpen && (
                    <div className="apple-menu-popover calendar-popover fade-in-scale">
                      <div className="menu-header">Select Recorded Date</div>
                      {availableDates.map((dateStr) => {
                        const isSelected = selectedDate === dateStr;
                        return (
                          <button
                            key={dateStr}
                            className={`menu-item ${isSelected ? 'selected' : ''}`}
                            onClick={() => handleSelectDate(dateStr)}
                          >
                            <div className="menu-item-text">
                              <span className="item-title">{formatDateDisplay(dateStr)}</span>
                              <span className="item-desc font-mono">{dateStr}</span>
                            </div>
                            {isSelected && <Check size={14} className="item-check" />}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {/* Status & Filter Metadata Bar */}
              <div className="clips-meta-bar">
                <div className="meta-left">
                  {activeSearch ? (
                    <span className="filter-summary">
                      Semantic Search for: <strong>"{activeSearch}"</strong> (
                      {clips.length} matches)
                    </span>
                  ) : (
                    <span className="filter-summary">
                      <strong>{formatDateDisplay(selectedDate)}</strong> — Showing{' '}
                      <strong>{clips.length}</strong> total events
                    </span>
                  )}
                </div>

                <div className="meta-right">
                  <span className="status-badge-mini processed">
                    <span className="mini-dot green" /> {totalProcessed} Processed
                  </span>
                  <span className="status-badge-mini unprocessed">
                    <span className="mini-dot amber" /> {totalUnprocessed} Unprocessed
                  </span>

                  {activeSearch && (
                    <button className="apple-link-btn" onClick={handleClearSearch}>
                      Reset search
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Clips Timeline Grid */}
            {loadingClips && clips.length === 0 ? (
              <div className="empty-state-card">
                <RefreshCw size={28} className="spin-animation empty-icon" />
                <h3 className="empty-title">Loading Footage Events...</h3>
              </div>
            ) : clips.length === 0 ? (
              <div className="empty-state-card">
                <FileVideo size={40} className="empty-icon" />
                <h3 className="empty-title">No Clips Available for {formatDateDisplay(selectedDate)}</h3>
                <p className="empty-subtitle">
                  Switch to another date with the calendar button or navigate to the{' '}
                  <button className="apple-inline-link" onClick={() => setActiveTab('ingest')}>
                    Studio & Ingest
                  </button>{' '}
                  tab to process new video footage.
                </p>
              </div>
            ) : (
              <div className="apple-clips-grid">
                {clips.map((clip) => {
                  const isProcessed = clip.status === 'processed';
                  const isAnalyzing = analyzingClipId === clip.clip_id;

                  return (
                    <div
                      key={clip.clip_id}
                      className={`apple-clip-card ${isProcessed ? 'is-processed' : 'is-unprocessed'}`}
                      onClick={() => setActiveModalClip(clip)}
                    >
                      {/* Media Thumbnail Container */}
                      <div className="card-media-box">
                        <img
                          src={clip.thumbnail_url}
                          alt={clip.title}
                          className="clip-thumb-img"
                          loading="lazy"
                          onError={(e) => {
                            // Fallback if image fails to render
                            (e.target as HTMLElement).style.display = 'none';
                          }}
                        />

                        {/* Play Hover Overlay */}
                        <div className="play-glass-overlay">
                          <div className="play-circle-icon">
                            <Play size={18} fill="#fff" />
                          </div>
                        </div>

                        {/* Top Badges */}
                        <div className="media-badge-top-left">
                          <span
                            className={`apple-status-pill ${
                              isProcessed ? 'pill-green' : 'pill-amber'
                            }`}
                          >
                            <span className="pill-dot" />
                            {isProcessed ? 'Processed' : 'Unprocessed'}
                          </span>
                        </div>

                        {clip.similarity !== undefined && clip.similarity !== null && (
                          <div className="media-badge-top-right">
                            <span
                              className={`match-pill ${
                                (clip.confidence_pct ?? Math.round(clip.similarity * 100)) >= 80
                                  ? 'match-high'
                                  : (clip.confidence_pct ?? Math.round(clip.similarity * 100)) >= 60
                                  ? 'match-med'
                                  : 'match-low'
                              }`}
                              title={`Google VideoPrism Raw Cosine: ${clip.similarity.toFixed(4)}`}
                            >
                              🎯 {clip.confidence_pct !== undefined && clip.confidence_pct !== null
                                ? `${Math.round(clip.confidence_pct)}% Match`
                                : `${Math.round(clip.similarity * 100)}% Match`}
                            </span>
                          </div>
                        )}

                        {/* Duration Bottom Right */}
                        <div className="media-badge-bottom-right">
                          <span className="duration-pill">{clip.duration_sec}s</span>
                        </div>
                      </div>

                      {/* Card Information */}
                      <div className="card-info-box">
                        <div className="info-header-line">
                          <span
                            className="timestamp-badge"
                            title={`${clip.timestamp || ''}${clip.time_range ? ` (${clip.time_range})` : ''}`}
                          >
                            <Clock size={12} />
                            {getDisplayTimestamp(clip)}
                          </span>
                          <span className="size-badge font-mono">{clip.size_mb} MB</span>
                        </div>

                        <h4 className="clip-card-title" title={clip.title}>
                          {clip.title}
                        </h4>

                        {/* Trajectory / Motion Caption */}
                        {clip.caption && (
                          <div className="clip-trajectory-badge">
                            <Compass size={13} color="#0a84ff" style={{ flexShrink: 0 }} />
                            <span className="trajectory-text">{clip.caption}</span>
                          </div>
                        )}

                        {/* Summary / Description */}
                        <p className="clip-description-preview">
                          {clip.action_summary ||
                            'Motion event extracted from footage. Awaiting Qwen3-VL analysis.'}
                        </p>

                        {/* Action Buttons Row */}
                        <div className="card-action-bar">
                          {isProcessed ? (
                            <button
                              className="apple-card-action-btn primary"
                              onClick={(e) => {
                                e.stopPropagation();
                                setActiveModalClip(clip);
                              }}
                            >
                              <Eye size={13} />
                              <span>Inspect Dossier</span>
                            </button>
                          ) : (
                            <button
                              className="apple-card-action-btn analyze-btn"
                              onClick={(e) => handleAnalyzeSingleClip(clip.clip_id, e)}
                              disabled={isAnalyzing}
                            >
                              {isAnalyzing ? (
                                <>
                                  <RefreshCw size={13} className="spin-animation" />
                                  <span>Analyzing VLM...</span>
                                </>
                              ) : (
                                <>
                                  <Zap size={13} color="#ff9f0a" />
                                  <span>⚡ Analyze Now</span>
                                </>
                              )}
                            </button>
                          )}

                          {isProcessed && (
                            <button
                              className="apple-card-icon-btn danger"
                              title="Delete from Vector DB"
                              onClick={(e) => handleDeleteClip(clip.clip_id, e)}
                            >
                              <Trash2 size={13} />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}

        {/* ================================================================ */}
        {/* TAB 3: SYSTEM DIAGNOSTICS & VECTOR DATABASE                      */}
        {/* ================================================================ */}
        {activeTab === 'system' && (
          <section className="tab-pane fade-in">
            <div className="section-hero">
              <h2 className="section-title">System Architecture & Hardware Diagnostics</h2>
              <p className="section-description">
                Real-time telemetry of NVIDIA CUDA GPU acceleration, Intel multicore compute,
                unified RAM utilization, and ChromaDB vector store statistics.
              </p>
            </div>

            <div className="system-overview-grid">
              {/* GPU Diagnostics Card */}
              <div className="apple-card">
                <div className="card-header">
                  <div className="card-title">
                    <Zap size={16} color="#30d158" />
                    <span>Graphics & CUDA Accelerator</span>
                  </div>
                  <span className="badge-subtle">
                    {systemStatus?.gpu.available ? 'CUDA 12.8 Active' : 'Offline'}
                  </span>
                </div>

                <div className="card-body">
                  <div className="spec-hero-name">
                    {systemStatus?.gpu.name || 'NVIDIA GeForce RTX 2050'}
                  </div>
                  <div className="spec-hero-sub">
                    {systemStatus?.gpu.device_capability || 'Ampere Architecture (4GB VRAM)'}
                  </div>

                  {/* VRAM Meter */}
                  <div className="meter-block">
                    <div className="meter-label-row">
                      <span>VRAM Allocation</span>
                      <span className="font-mono">
                        {systemStatus?.gpu.vram_allocated_gb ?? 0} GB /{' '}
                        {systemStatus?.gpu.vram_total_gb ?? 4.0} GB (
                        {systemStatus?.gpu.vram_percent ?? 0}%)
                      </span>
                    </div>
                    <div className="apple-meter-track">
                      <div
                        className="apple-meter-fill fill-green"
                        style={{ width: `${Math.min(100, systemStatus?.gpu.vram_percent ?? 0)}%` }}
                      />
                    </div>
                    <div className="meter-subinfo">
                      <span>Available Free VRAM:</span>
                      <strong className="font-mono">
                        {systemStatus?.gpu.vram_free_gb ?? 0} GB Free
                      </strong>
                    </div>
                  </div>
                </div>
              </div>

              {/* CPU Diagnostics Card */}
              <div className="apple-card">
                <div className="card-header">
                  <div className="card-title">
                    <Cpu size={16} color="#0a84ff" />
                    <span>Host CPU & Compute Cores</span>
                  </div>
                  <span className="badge-subtle font-mono">
                    {systemStatus?.cpu.architecture || 'AMD64'}
                  </span>
                </div>

                <div className="card-body">
                  <div className="spec-hero-name">
                    {systemStatus?.cpu.name || '11th Gen Intel(R) Core(TM) i5-11400H'}
                  </div>
                  <div className="spec-hero-sub">
                    {systemStatus?.cpu.cores ?? 6} Physical Cores /{' '}
                    {systemStatus?.cpu.threads ?? 12} Logical Compute Threads
                  </div>

                  {/* CPU Usage Meter */}
                  <div className="meter-block">
                    <div className="meter-label-row">
                      <span>Compute Load</span>
                      <span className="font-mono">{systemStatus?.cpu.usage_pct ?? 0}%</span>
                    </div>
                    <div className="apple-meter-track">
                      <div
                        className="apple-meter-fill fill-blue"
                        style={{ width: `${Math.min(100, systemStatus?.cpu.usage_pct ?? 0)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* System RAM Card */}
              <div className="apple-card">
                <div className="card-header">
                  <div className="card-title">
                    <HardDrive size={16} color="#bf5af2" />
                    <span>Unified System Memory (RAM)</span>
                  </div>
                  <span className="badge-subtle font-mono">
                    {systemStatus?.ram.total_gb ?? 16} GB DDR4
                  </span>
                </div>

                <div className="card-body">
                  <div className="spec-hero-name">DDR4 High-Speed RAM</div>
                  <div className="spec-hero-sub">
                    Used: {systemStatus?.ram.used_gb ?? 0} GB • Free:{' '}
                    {systemStatus?.ram.free_gb ?? 0} GB
                  </div>

                  {/* RAM Meter */}
                  <div className="meter-block">
                    <div className="meter-label-row">
                      <span>Memory In Use</span>
                      <span className="font-mono">{systemStatus?.ram.percent ?? 0}%</span>
                    </div>
                    <div className="apple-meter-track">
                      <div
                        className="apple-meter-fill fill-purple"
                        style={{ width: `${Math.min(100, systemStatus?.ram.percent ?? 0)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Vector DB (ChromaDB) Card */}
              <div className="apple-card">
                <div className="card-header">
                  <div className="card-title">
                    <Database size={16} color="#ff9f0a" />
                    <span>ChromaDB Vector Store</span>
                  </div>
                  <span className="badge-subtle">HNSW + Cosine</span>
                </div>

                <div className="card-body">
                  <div className="spec-hero-name">
                    Collection: {systemStatus?.vector_db.collection || 'cctv_motion_events'}
                  </div>
                  <div className="spec-hero-sub font-mono">
                    Path: {systemStatus?.vector_db.storage_path || 'data/chroma_db'}
                  </div>

                  <div className="stat-pills-row">
                    <div className="stat-pill-box">
                      <span className="stat-label">Indexed in DB</span>
                      <span className="stat-value font-mono">
                        {systemStatus?.vector_db.total_indexed_clips ?? 0}
                      </span>
                    </div>

                    <div className="stat-pill-box">
                      <span className="stat-label">Clips on Disk</span>
                      <span className="stat-value font-mono">
                        {systemStatus?.vector_db.total_disk_clips ?? 0}
                      </span>
                    </div>

                    <div className="stat-pill-box">
                      <span className="stat-label">Pending VLM</span>
                      <span className="stat-value font-mono">
                        {systemStatus?.vector_db.unprocessed_clips ?? 0}
                      </span>
                    </div>
                  </div>

                  <div className="db-actions-row">
                    <button className="apple-danger-btn small" onClick={handleResetDB}>
                      <Trash2 size={13} />
                      <span>Reset Vector DB</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Architecture & Model Specifications */}
            <div className="apple-card" style={{ marginTop: '24px' }}>
              <div className="card-header">
                <div className="card-title">
                  <Sparkles size={16} color="#0a84ff" />
                  <span>Multimodal Neural Models & Quantization</span>
                </div>
                <span className="badge-subtle">Local Offline Execution</span>
              </div>

              <div className="card-body">
                <div className="models-comparison-grid">
                  <div className="model-spec-card">
                    <div className="model-card-tag">Vision-Language Model</div>
                    <h3 className="model-name">
                      {systemStatus?.vlm_model.name || 'Qwen3-VL-4B-Instruct'}
                    </h3>
                    <ul className="model-spec-list">
                      <li>
                        <strong>Quantization:</strong>{' '}
                        {systemStatus?.vlm_model.quantization || '4-bit NF4 (BitsAndBytes)'}
                      </li>
                      <li>
                        <strong>Attention Kernel:</strong>{' '}
                        {systemStatus?.vlm_model.attention || 'Compiled SDPA / FlashAttention'}
                      </li>
                      <li>
                        <strong>Inference Hardware:</strong>{' '}
                        {systemStatus?.vlm_model.device || 'NVIDIA RTX 2050 (Hybrid Overflow)'}
                      </li>
                      <li>
                        <strong>Incident Memory:</strong>{' '}
                        {systemStatus?.vlm_model.context_tracking || 'Chronological Event Ledger & Entity Re-ID'}
                      </li>
                      <li>
                        <strong>Output Schema:</strong> Structured Title, Motion Trajectory, &
                        Hierarchical Action Summary
                      </li>
                    </ul>
                  </div>

                  <div className="model-spec-card">
                    <div className="model-card-tag">Multimodal Video Embedder</div>
                    <h3 className="model-name">
                      {systemStatus?.embedding_model.name || 'Google VideoPrism (videoprism_lvt_public_v1_base)'}
                    </h3>
                    <ul className="model-spec-list">
                      <li>
                        <strong>Embedding Dimension:</strong>{' '}
                        {systemStatus?.embedding_model.dimensions || 768} Dense Floats
                      </li>
                      <li>
                        <strong>Temporal Window:</strong>{' '}
                        16 Frames @ 288×288 RGB
                      </li>
                      <li>
                        <strong>Inference Hardware:</strong>{' '}
                        {systemStatus?.embedding_model.device || 'Host CPU / CUDA (0% VRAM Contention)'}
                      </li>
                      <li>
                        <strong>Vector Metric:</strong> Cosine Similarity via HNSW Graph (ChromaDB)
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* ================================================================== */}
      {/* Interactive Video & Forensic Dossier Modal                         */}
      {/* ================================================================== */}
      {activeModalClip && (
        <div className="apple-modal-backdrop" onClick={() => setActiveModalClip(null)}>
          <div className="apple-modal-sheet" onClick={(e) => e.stopPropagation()}>
            <div className="modal-sheet-header">
              <div className="modal-sheet-title-group">
                <h3 className="modal-sheet-title">{activeModalClip.title}</h3>
                <span className="modal-sheet-sub">
                  Camera: {activeModalClip.camera_id} • Duration: {activeModalClip.duration_sec}s
                </span>
              </div>
              <button
                className="modal-close-icon-btn"
                onClick={() => setActiveModalClip(null)}
                title="Close"
              >
                <X size={18} />
              </button>
            </div>

            <div className="modal-sheet-body">
              {/* Left Column: Player */}
              <div className="modal-player-col">
                <div className="video-viewport">
                  <video
                    src={activeModalClip.video_url}
                    controls
                    autoPlay
                    loop
                    playsInline
                  />
                </div>

                <div className="video-stats-strip">
                  <div className="stat-item">
                    <span>Timestamp</span>
                    <strong>{activeModalClip.timestamp}</strong>
                  </div>
                  <div className="stat-item">
                    <span>File Size</span>
                    <strong>{activeModalClip.size_mb} MB</strong>
                  </div>
                  <div className="stat-item">
                    <span>Status</span>
                    <strong style={{ color: activeModalClip.status === 'processed' ? '#30d158' : '#ff9f0a' }}>
                      {activeModalClip.status.toUpperCase()}
                    </strong>
                  </div>
                </div>
              </div>

              {/* Right Column: Dossier */}
              <div className="modal-dossier-col">
                {activeModalClip.status === 'unprocessed' ? (
                  <div className="unprocessed-dossier-notice">
                    <Info size={28} color="#ff9f0a" />
                    <h4>Unprocessed Motion Event</h4>
                    <p>
                      This clip was automatically extracted by hardware NVDEC motion analysis. It
                      has not yet been analyzed by Qwen3-VL.
                    </p>
                    <button
                      className="apple-primary-btn"
                      onClick={(e) => {
                        handleAnalyzeSingleClip(activeModalClip.clip_id, e);
                        setActiveModalClip(null);
                      }}
                      disabled={analyzingClipId === activeModalClip.clip_id}
                    >
                      <Zap size={15} />
                      <span>⚡ Analyze with Qwen3-VL Now</span>
                    </button>
                  </div>
                ) : (
                  <>
                    {activeModalClip.caption && (
                      <div className="dossier-field">
                        <span className="dossier-field-label">Motion Trajectory & Vector</span>
                        <div className="dossier-field-value">{activeModalClip.caption}</div>
                      </div>
                    )}

                    {activeModalClip.action_summary && (
                      <div className="dossier-field">
                        <span className="dossier-field-label">Executive Forensic Summary</span>
                        <div className="dossier-field-value">{activeModalClip.action_summary}</div>
                      </div>
                    )}

                    {activeModalClip.full_report && (
                      <div className="dossier-field">
                        <span className="dossier-field-label">
                          Full Qwen3-VL Forensic Dossier (Indexed in ChromaDB)
                        </span>
                        <pre className="dossier-pre-block">{activeModalClip.full_report}</pre>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
