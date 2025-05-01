# AI-Powered Strategic Intelligence Assistant

An autonomous agent capable of browsing the web (or a simulated database) to collect public information about competitors, synthesize it, and suggest strategic differentiation or repositioning opportunities.

## Features

- Complete SWOT Analysis
- PESTEL Analysis
- Porter's Five Forces Analysis
- BCG Matrix
- McKinsey 7S Model
- Competitor Analysis
- Automated LinkedIn Analysis
- Visualization Generation
- Document (TXT for the moment, pdf and other type will be updated soon) Support for RAG Analysis

## Architecture

The project is structured into two main components:

### Backend (FastAPI)
- RESTful API for business analysis
- OpenAI and Tavily integration
- Automated web scraping
- Visualization generation
- RAG support for document analysis

### Frontend (React + TypeScript)
- Modern UI with Tailwind CSS
- Interactive visualizations
- Form management
- Analysis results display

## Prerequisites

- Docker

### Configuration Files
Before starting, you need to create a `.env` file in the project root directory with the following environment variables:

```bash
# API Keys
TAVILY_API_KEY=your_tavily_key
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1

# LinkedIn Configuration
LINKEDIN_EMAIL=your_linkedin_email
LINKEDIN_PASSWORD=your_linkedin_password
```

## Docker Installation

1. Clone the repository:
```bash
git clone https://github.com/kcap02DVT/ghost_analysis.git

cd ghost_analysis

```

2. Create the `.env` file with your API keys:
```bash
cp .env.example .env
# Edit the .env file with your API keys
```

3. Build and start the containers:
```bash
# Build the images
docker-compose build

# Start the services
docker-compose up -d
```

4. Verify that containers are running:
```bash
docker-compose ps
```

5. Access the application:
- Frontend: http://localhost:8080
- Backend API: http://localhost:8080/analyze


## Technologies Used

### Backend
- FastAPI
- Uvicorn
- LangChain
- OpenAI API
- Tavily Search API
- Selenium
- BeautifulSoup
- Matplotlib

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS
- Headless UI
- Lucide Icons



## Contact

For any questions or suggestions, please open an issue on GitHub. 
