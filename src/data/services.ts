import { Service } from '../types';

export const services: Service[] = [
  {
    id: 1,
    title: 'Automated Competitive Monitoring',
    description: 'Gathering of public information on competitors (websites, news, social media, simulated internal databases)',
    icon: 'layers',
    details: {
      overview: 'Our custom blockchain development services provide end-to-end solutions tailored to your specific business requirements. We build secure, scalable, and efficient blockchain networks that can transform your operations.',
      benefits: [
        'Improved security and transparency',
        'Enhanced efficiency and reduced costs',
        'Customized to your specific business requirements',
        'Scalable architecture for future growth'
      ],
      features: [
        'Private and consortium blockchain networks',
        'Smart contract development and auditing',
        'Consensus mechanism customization',
        'Integration with existing systems'
      ]
    }
  },
  {
    id: 2,
    title: 'Differentiation Strategy',
    description: 'Analysis of collected information to provide strategic recommendations (pricing, product, positioning, innovation)',
    icon: 'users',
    details: {
      overview: 'Our blockchain consulting services help businesses understand how blockchain can solve their problems. We provide strategic guidance, technical expertise, and implementation roadmaps to ensure successful blockchain adoption.',
      benefits: [
        'Clear blockchain adoption strategy',
        'Risk assessment and mitigation',
        'Competitive advantage in your industry',
        'Expert guidance from industry specialists'
      ],
      features: [
        'Blockchain feasibility assessment',
        'Technology selection and architecture planning',
        'Implementation roadmap development',
        'Staff training and knowledge transfer'
      ]
    }
  },
  {
    id: 3,
    title: 'Strategic Document Generation',
    description: 'Automated creation of structured documents (such as SWOT, Porter’s Five Forces, BCG Matrix, etc.) ready for reporting use',
    icon: 'database',
    details: {
      overview: 'Our enterprise blockchain solutions are designed to meet the complex needs of large organizations. We develop high-performance, secure, and compliant blockchain systems that integrate seamlessly with your enterprise architecture.',
      benefits: [
        'Enhanced data integrity and security',
        'Streamlined business processes',
        'Reduced operational costs',
        'Improved compliance and auditability'
      ],
      features: [
        'Enterprise-grade security and performance',
        'Seamless integration with legacy systems',
        'Compliance with industry regulations',
        'High throughput and low latency solutions'
      ]
    }
  }
];