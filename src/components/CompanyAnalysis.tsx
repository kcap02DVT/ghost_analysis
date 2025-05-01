import React, { useState } from 'react';

interface CompanyAnalysisProps {
  isVisible: boolean;
  onAnalysisComplete?: (result: AnalysisResult | null) => void;
}

interface AnalysisResult {
  company_name: string;
  summaries: string[];
  strategy_recommendations: string;
  swot_lists: {
    strengths: string[];
    weaknesses: string[];
    opportunities: string[];
    threats: string[];
  };
  swot_image: string;
}

const CompanyAnalysis: React.FC<CompanyAnalysisProps> = ({ isVisible, onAnalysisComplete }) => {
  const [companyName, setCompanyName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!companyName.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('company_name', companyName);
      if (file) {
        formData.append('file', file);
      }

      const response = await fetch('https://api-hackaton-123787782603.europe-west9.run.app//analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze company');
      }

      const data = await response.json();
      if (onAnalysisComplete) {
        onAnalysisComplete(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      if (onAnalysisComplete) {
        onAnalysisComplete(null);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  if (!isVisible) return null;

  return (
    <div className="mt-8 space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex flex-col gap-2">
          <label htmlFor="file-upload" className="text-white/80">
            Upload Document
          </label>
          <div className="relative">
            <input
              id="file-upload"
              type="file"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              accept=".txt"
            />
            <div className="px-4 py-2 rounded-lg bg-white/10 text-white border border-white/20 flex items-center">
              <span className="bg-blue-500 text-white px-4 py-2 rounded-lg mr-4 text-sm font-semibold hover:bg-blue-600">
                Choose File
              </span>
              <span className="text-gray-400">
                {file ? file.name : 'No file selected'}
              </span>
            </div>
          </div>
        </div>
        
        <div className="flex gap-4">
          <input
            type="text"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="Enter company name..."
            className="flex-1 px-4 py-2 rounded-lg bg-white/10 text-white placeholder-gray-400 border border-white/20 focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 disabled:opacity-50"
          >
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>
      </form>

      {error && (
        <div className="p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200">
          {error}
        </div>
      )}
    </div>
  );
};

export default CompanyAnalysis; 