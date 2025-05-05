export interface Service {
  id: number;
  title: string;
  description: string;
  icon: string;
  details: {
    overview: string;
    benefits: string[];
    features: string[];
  };
}

export interface AnalysisResult {
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
  porter_forces: {
    rivalry: string[];
    new_entrants: string[];
    substitutes: string[];
    buyer_power: string[];
    supplier_power: string[];
  };
  porter_image: string;
  bcg_matrix: {
    [key: string]: {
      market_share: number;
      growth_rate: number;
    };
  };
  bcg_image: string;
  mckinsey_7s: {
    strategy: string;
    structure: string;
    systems: string;
    style: string;
    staff: string;
    skills: string;
    shared_values: string;
  };
  mckinsey_image: string;
  sources: string[];
  linkedin_analysis?: string;
}