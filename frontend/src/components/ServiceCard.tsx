import React from 'react';
import * as LucideIcons from 'lucide-react';
import { Service } from '../types';

interface ServiceCardProps {
  service: Service;
  onClick: () => void;
}

const ServiceCard: React.FC<ServiceCardProps> = ({ service, onClick }) => {
  // Dynamically get the icon from lucide-react
  const IconComponent = (LucideIcons as Record<string, React.FC<any>>)[
    service.icon.charAt(0).toUpperCase() + service.icon.slice(1)
  ] || LucideIcons.CircleDot;

  return (
    <div 
      className="bg-purple-800/30 backdrop-blur-sm rounded-2xl p-6 flex flex-col items-center transition-all duration-300 group cursor-pointer hover:bg-purple-800/40"
      onClick={onClick}
    >
      <div className="relative mb-4">
        {/* Glow effect behind icon */}
        <div className="absolute inset-0 bg-blue-500 opacity-20 rounded-full blur-xl group-hover:opacity-30 transition-opacity"></div>
        
        {/* Icon circle */}
        <div className="w-24 h-24 rounded-full bg-gradient-to-br from-blue-400 to-blue-600 flex items-center justify-center relative z-10 transition-transform group-hover:scale-105 shadow-lg shadow-blue-900/30">
          <IconComponent className="w-10 h-10 text-white" />
        </div>
      </div>
      
      <h3 className="text-white text-center font-semibold text-lg mb-2">{service.title}</h3>
      <p className="text-gray-300 text-center text-sm mb-6">{service.description}</p>
      
      <div className="mt-auto">
        <button 
          className="bg-gradient-to-r from-cyan-400 to-blue-500 text-white px-5 py-2 rounded-full text-sm font-medium transition-transform hover:scale-105 shadow-lg shadow-blue-900/20"
        >
          Learn More
        </button>
      </div>
    </div>
  );
};

export default ServiceCard;