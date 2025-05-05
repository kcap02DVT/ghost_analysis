import React from 'react';
import CompanyAnalysis from './CompanyAnalysis';

interface HeaderProps {
  onAnalysisComplete: (result: any) => void;
}

const Header: React.FC<HeaderProps> = ({ onAnalysisComplete }) => {
  return (
    <div className="text-center mb-16">
      <h1 className="text-4xl md:text-5xl font-bold text-black mb-4">Ghost Agent</h1>
      <p className="text-black-300 max-w-2xl mx-auto mb-8">
      As a leading provider of AI-powered competitive intelligence solutions, we deliver cutting-edge tools that autonomously gather, synthesize, and transform market insights into actionable strategic opportunities.
      </p>
      
      {/* Company Analysis Component */}
      <div className="max-w-2xl mx-auto">
        <CompanyAnalysis isVisible={true} onAnalysisComplete={onAnalysisComplete} />
      </div>
    </div>
  );
};

export default Header;