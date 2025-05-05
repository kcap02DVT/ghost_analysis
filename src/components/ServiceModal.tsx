import React from 'react';
import { X } from 'lucide-react';
import { Service } from '../types';

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
  pestel_lists: {
    political: string[];
    economic: string[];
    social: string[];
    technological: string[];
    environmental: string[];
    legal: string[];
  };
  pestel_image: string;
  sources: string[];
  linkedin_analysis: string;
  porter_image: string;
  porter_forces: Record<string, string[]>;
  bcg_image: string;
  bcg_matrix: Record<string, { market_share: string; growth_rate: string }>;
  mckinsey_image: string;
  mckinsey_7s: Record<string, string>;
}

interface ServiceModalProps {
  service: Service | null;
  isOpen: boolean;
  onClose: () => void;
  analysisResult: AnalysisResult | null;
}

const ServiceModal: React.FC<ServiceModalProps> = ({ service, isOpen, onClose, analysisResult }) => {
  if (!service || !isOpen) return null;

  const isStrategyService = service.title === "Differentiation Strategy";
  const isDocumentService = service.title === "Strategic Document Generation";
  const isCompetitiveAnalysisService = service.title === "Automated Competitive Monitoring";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm" 
        onClick={onClose}
      ></div>
      
      {/* Modal */}
      <div 
        className="bg-purple-800/30 backdrop-blur-sm rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-auto relative z-10 shadow-xl shadow-blue-900/20"
        style={{ 
          animation: isOpen ? 'modalFadeIn 0.3s ease-out forwards' : 'none',
        }}
      >
        {/* Close button */}
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-black p-1 rounded-full hover:bg-white/10 transition-colors"
        >
          <X size={20} />
        </button>
        
        {/* Content */}
        <div className="p-8">
          <h2 className="text-2xl font-bold text-black mb-4">{service.title}</h2>
          
          {/* Strategic Recommendations */}
          {isStrategyService && analysisResult && (
            <div className="mt-8 space-y-6 bg-white/5 p-6 rounded-xl">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-2">Differentiation Strategy for {analysisResult.company_name}</h3>
                <p 
                  className="text-gray-300"
                  dangerouslySetInnerHTML={{ __html: analysisResult.strategy_recommendations }}
                />
              </div>
            </div>
          )}

          {/* Competitive Analysis */}
          {isCompetitiveAnalysisService && analysisResult && (
            <div className="mt-8 space-y-6">
              {/* LinkedIn Analysis */}
              {analysisResult.linkedin_analysis && (
                <div className="bg-white/5 p-6 rounded-xl">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">LinkedIn Analysis</h3>
                  <div 
                    className="text-gray-300"
                    dangerouslySetInnerHTML={{ __html: analysisResult.linkedin_analysis }}
                  />
                </div>
              )}

              {/* Web Analysis */}
              {analysisResult.summaries && analysisResult.summaries.length > 0 && (
                <div className="bg-white/5 p-6 rounded-xl">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Competitive Monitoring for {analysisResult.company_name}</h3>
                  <div className="space-y-4">
                    {analysisResult.summaries.map((summary, index) => (
                      <div key={index} className="p-4 bg-white/5 rounded-lg">
                        <p 
                          className="text-gray-300"
                          dangerouslySetInnerHTML={{ __html: summary }}
                        />
                        {analysisResult.sources && analysisResult.sources[index] && (
                          <a
                            href={analysisResult.sources[index]}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-block mt-2 text-sm text-blue-400 hover:text-blue-300 transition-colors"
                          >
                            Source: {new URL(analysisResult.sources[index]).hostname}
                          </a>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* SWOT and PESTEL Analysis */}
          {isDocumentService && analysisResult && (
            <div className="mt-8 space-y-8">
              {/* SWOT Analysis */}
              <div className="space-y-6 bg-white/5 p-6 rounded-xl">
                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-2">SWOT Analysis for {analysisResult.company_name}</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <h4 className="font-medium text-green-400">Strengths</h4>
                      <ul className="list-disc list-inside text-gray-300">
                        {analysisResult.swot_lists.strengths.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium text-red-400">Weaknesses</h4>
                      <ul className="list-disc list-inside text-gray-300">
                        {analysisResult.swot_lists.weaknesses.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium text-yellow-400">Opportunities</h4>
                      <ul className="list-disc list-inside text-gray-300">
                        {analysisResult.swot_lists.opportunities.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="space-y-2">
                      <h4 className="font-medium text-purple-400">Threats</h4>
                      <ul className="list-disc list-inside text-gray-300">
                        {analysisResult.swot_lists.threats.map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-2">SWOT Visualization</h3>
                  <img
                    src={`data:image/png;base64,${analysisResult.swot_image}`}
                    alt="SWOT Analysis"
                    className="w-full rounded-lg"
                  />
                </div>
              </div>

              {/* PESTEL Analysis */}
              <div className="space-y-6 bg-white/5 p-6 rounded-xl">
                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-2">PESTEL Analysis for {analysisResult.company_name}</h3>
                  <img
                    src={`data:image/png;base64,${analysisResult.pestel_image}`}
                    alt="PESTEL Analysis"
                    className="w-full rounded-lg"
                  />
                </div>
              </div>

              {/* Porter's Five Forces */}
              <div className="space-y-6 bg-white/5 p-6 rounded-xl">
                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-2">Porter's Five Forces Analysis</h3>
                  <img
                    src={`data:image/png;base64,${analysisResult.porter_image}`}
                    alt="Porter's Five Forces"
                    className="w-full rounded-lg"
                  />
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    {Object.entries(analysisResult.porter_forces).map(([force, items]) => (
                      <div key={force} className="space-y-2">
                        <h4 className="font-medium text-blue-400 capitalize">{force.replace('_', ' ')}</h4>
                        <ul className="list-disc list-inside text-gray-300">
                          {items.map((item, i) => (
                            <li key={i}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* BCG Matrix */}
              <div className="space-y-6 bg-white/5 p-6 rounded-xl">
                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-2">BCG Matrix Analysis</h3>
                  <img
                    src={`data:image/png;base64,${analysisResult.bcg_image}`}
                    alt="BCG Matrix"
                    className="w-full rounded-lg"
                  />
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    {Object.entries(analysisResult.bcg_matrix).map(([product, values]) => (
                      <div key={product} className="p-3 bg-white/5 rounded-lg">
                        <h4 className="font-medium text-blue-400">{product}</h4>
                        <p className="text-gray-300">Market Share: {values.market_share}</p>
                        <p className="text-gray-300">Growth Rate: {values.growth_rate}%</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* McKinsey 7S */}
              <div className="space-y-6 bg-white/5 p-6 rounded-xl">
                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-2">McKinsey 7S Model</h3>
                  <img
                    src={`data:image/png;base64,${analysisResult.mckinsey_image}`}
                    alt="McKinsey 7S Model"
                    className="w-full rounded-lg"
                  />
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    {Object.entries(analysisResult.mckinsey_7s).map(([element, value]) => (
                      <div key={element} className="p-3 bg-white/5 rounded-lg">
                        <h4 className="font-medium text-blue-400 capitalize">{element.replace('_', ' ')}</h4>
                        <p className="text-gray-300">{value}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div className="mt-8">
        
          </div>
        </div>
      </div>
    </div>
  );
};

export default ServiceModal;