import React, { useState } from 'react';
import Header from './components/Header';
import ServiceCard from './components/ServiceCard';
import ServiceModal from './components/ServiceModal';
import { services } from './data/services';
import { Service } from './types';

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
  pestel_image: string;
  pestel_lists?: {
    political: string[];
    economic: string[];
    social: string[];
    technological: string[];
    environmental: string[];
    legal: string[];
  };
}

function App() {
  const [selectedService, setSelectedService] = useState<Service | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);

  const handleCardClick = (service: Service) => {
    setSelectedService(service);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setTimeout(() => setSelectedService(null), 300); // Clear after animation
  };

  const handleAnalysisComplete = (result: AnalysisResult | null) => {
    setAnalysisResult(result);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-950 to-purple-900 text-white flex flex-col">
      <div className="container mx-auto px-4 py-16 flex-grow">
        <Header onAnalysisComplete={handleAnalysisComplete} />
        
        {/* Service Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          {services.map((service) => (
            <ServiceCard
              key={service.id}
              service={service}
              onClick={() => handleCardClick(service)}
            />
          ))}
        </div>

        {/* Modal */}
        <ServiceModal 
          service={selectedService}
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          analysisResult={analysisResult}
        />
      </div>
      
      {/* Footer */}
      <footer className="w-full py-4 text-center text-white/70 border-t border-white/10">
        Powered by Mistral
      </footer>
    </div>
  );
}

export default App;