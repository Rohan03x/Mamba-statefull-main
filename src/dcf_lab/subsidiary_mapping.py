"""
Subsidiary Mapping System for Corporate Structure Analysis

This module implements subsidiary/segment mapping from SEC Exhibit 21 filings
to build corporate trees and propagate risk/revenue exposures across entities.

Key Features:
- Parse SEC Exhibit 21 subsidiary lists from 10-K/10-Q filings
- Build corporate ownership hierarchies
- Calculate subsidiary risk exposure weights
- Map geographic and business line exposures
- Integration with GLEIF LEI data for entity validation

Data Sources:
- SEC EDGAR filings (Exhibit 21)
- GLEIF Level-2 Who-owns-Whom data
- Company segment reporting
"""

import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Constants
UNITED_STATES = 'United States'
CAYMAN_ISLANDS = 'Cayman Islands'
BRITISH_VIRGIN_ISLANDS = 'British Virgin Islands'


@dataclass
class SubsidiaryEntity:
    """Represents a subsidiary entity"""
    name: str
    jurisdiction: str
    ownership_percentage: float = 100.0
    business_segment: Optional[str] = None
    revenue_contribution: Optional[float] = None
    lei_code: Optional[str] = None
    ticker: Optional[str] = None
    parent_entity: Optional[str] = None
    entity_type: str = "subsidiary"  # subsidiary, joint_venture, investment


@dataclass
class CorporateStructure:
    """Represents the complete corporate structure"""
    parent_ticker: str
    entities: Dict[str, SubsidiaryEntity] = field(default_factory=dict)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    risk_weights: Dict[str, float] = field(default_factory=dict)
    geographic_exposure: Dict[str, float] = field(default_factory=dict)
    segment_exposure: Dict[str, float] = field(default_factory=dict)
    last_updated: Optional[str] = None


class SubsidiaryMapper:
    """
    Corporate Subsidiary Mapping System
    
    Extracts and analyzes subsidiary structures from SEC filings
    to understand corporate risk exposure and revenue attribution.
    """
    
    def __init__(self):
        self.sec_base_url = "https://www.sec.gov/Archives/edgar/data"
        self.gleif_base_url = "https://api.gleif.org/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DCF Lab Research Tool/1.0 (research@dcflab.com)',
            'Accept': 'application/json, text/html, application/xml'
        })
        
        # Common patterns for parsing subsidiary data
        self.ownership_patterns = [
            r'(\d+(?:\.\d+)?)%',  # Direct percentage
            r'(\d+(?:\.\d+)?)\s*percent',  # Percent spelled out
            r'wholly[- ]owned',  # 100% owned
            # Approximate percentages
            r'((?:approximately\s+)?(?:\d+(?:\.\d+)?))%'
        ]

        self.jurisdiction_patterns = [
            r'\b([A-Z]{2})\b',  # Country codes
            r'\b(United States?|USA?|US)\b',
            r'\b(Delaware|Nevada|California|New York)\b',  # US states
            r'\b(Cayman Islands?|British Virgin Islands?|Luxembourg|Ireland)\b'
        ]

    def get_corporate_structure(self, ticker: str) -> CorporateStructure:
        """
        Get corporate structure from EODHD fundamentals data
        
        Uses:
        - General.Listings: International exchange listings (proxy for geographic presence)
        - General.FullTimeEmployees: Employee count (proxy for organization complexity)
        - General.InternationalDomestic: International/Domestic classification
        
        Args:
            ticker: Stock ticker symbol

        Returns:
            CorporateStructure with estimated subsidiaries based on EODHD data
        """
        try:
            logger.info(f"🏢 Building corporate structure for {ticker} from EODHD")
            
            # Initialize structure
            structure = CorporateStructure(parent_ticker=ticker)
            
            # Get EODHD fundamentals data
            try:
                from src.data_sources.eodhd_provider import EODHDProvider
                provider = EODHDProvider()
                data = provider.get_fundamentals(ticker)
            except Exception as e:
                logger.error(f"Failed to get EODHD fundamentals: {e}")
                return structure
            
            if not data or 'General' not in data:
                logger.warning(f"No EODHD fundamentals data for {ticker}")
                return structure
            
            # Parse EODHD data to estimate subsidiaries
            general = data['General']
            subsidiary_data = self._parse_eodhd_listings(general, ticker)
            
            if subsidiary_data:
                structure.entities.update(subsidiary_data)
            
            if not structure.entities:
                msg = f"No subsidiary data derived from EODHD for {ticker}"
                logger.warning(msg)
                return structure

            # Build hierarchy and calculate exposures
            structure.hierarchy = self._build_hierarchy(structure.entities)
            structure.risk_weights = self._calculate_risk_weights(
                structure.entities)
            structure.geographic_exposure = (
                self._calculate_geographic_exposure(structure.entities))
            structure.segment_exposure = self._calculate_segment_exposure(
                structure.entities)

            structure.last_updated = datetime.now().strftime('%Y-%m-%d')

            entities_count = len(structure.entities)
            logger.info(f"✅ Built structure with {entities_count} entities from EODHD")
            return structure

        except Exception as e:
            msg = f"❌ Error building corporate structure for {ticker}: {e}"
            logger.error(msg)
            return CorporateStructure(parent_ticker=ticker)
    
    def _get_recent_filings(self, ticker: str) -> List[Dict[str, Any]]:
        """Get recent SEC filings for a ticker"""
        try:
            # SEC Company Tickers API
            tickers_url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(tickers_url)
            
            if response.status_code != 200:
                logger.warning("Could not fetch SEC ticker data")
                return []
            
            company_data = response.json()
            
            # Find CIK for ticker
            cik = None
            for entry in company_data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    break
            
            if not cik:
                logger.warning(f"Could not find CIK for ticker {ticker}")
                return self._get_filings_fallback()

            # Get recent filings
            submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.session.get(submissions_url)

            if response.status_code != 200:
                return self._get_filings_fallback()

            submissions = response.json()
            recent_filings = submissions.get('filings', {}).get('recent', {})

            filings = []
            forms = recent_filings.get('form', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            filing_dates = recent_filings.get('filingDate', [])

            for i, form in enumerate(forms):
                if form in ['10-K', '10-Q', '10-K/A']:
                    filings.append({
                        'form': form,
                        'accession_number': accession_numbers[i],
                        'filing_date': filing_dates[i],
                        'cik': cik
                    })

            # Sort by filing date (most recent first)
            filings.sort(key=lambda x: x['filing_date'], reverse=True)

            return filings[:5]  # Return last 5 filings

        except Exception as e:
            logger.warning(f"Error fetching SEC filings: {e}")
            return self._get_filings_fallback()
    
    def _get_filings_fallback(self) -> List[Dict[str, Any]]:
        """Fallback method to get filings when main API fails"""
        # MOCK DATA DISABLED
        logger.error("Mock filing data disabled")
        return []
        
        if False:  # Disabled
            return [{
            'form': '10-K',
            'accession_number': '0000000000-00-000000',
            'filing_date': datetime.now().strftime('%Y-%m-%d'),
            'cik': '0000000000'
        }]
    
    def _parse_eodhd_listings(
            self, general: Dict[str, Any], ticker: str) -> Dict[str, SubsidiaryEntity]:
        """
        Parse EODHD fundamentals to estimate subsidiary structure
        
        Uses:
        - Listings: International exchange listings → estimate geographic subsidiaries
        - FullTimeEmployees: Employee count → estimate organizational complexity
        - InternationalDomestic: International/Domestic classification
        - GicSector/Industry: Business segment classification
        
        Args:
            general: EODHD General section data
            ticker: Parent company ticker
            
        Returns:
            Dictionary of estimated subsidiary entities
        """
        try:
            subsidiaries = {}
            
            # 1. Create entities from international listings
            listings = general.get('Listings', {})
            if listings:
                for idx, listing in listings.items():
                    exchange = listing.get('Exchange', 'Unknown')
                    code = listing.get('Code', '')
                    name = listing.get('Name', f'{ticker} International')
                    
                    # Map exchange to country/jurisdiction
                    jurisdiction_map = {
                        'LSE': 'United Kingdom',  # London
                        'BA': 'Argentina',        # Buenos Aires
                        'SA': 'Brazil',           # Sao Paulo
                        'XETRA': 'Germany',       # Frankfurt
                        'PA': 'France',           # Paris
                        'TO': 'Canada',           # Toronto
                        'HK': 'Hong Kong',        # Hong Kong
                        'JP': 'Japan',            # Tokyo
                        'SG': 'Singapore',        # Singapore
                        'AU': 'Australia',        # Sydney
                    }
                    
                    jurisdiction = jurisdiction_map.get(exchange, 'International')
                    
                    entity_id = f"{ticker}_{exchange}_{code}"
                    subsidiaries[entity_id] = SubsidiaryEntity(
                        name=name,
                        jurisdiction=jurisdiction,
                        ownership_percentage=100.0,  # Assume full ownership of listings
                        business_segment=general.get('GicSector', 'Operations'),
                        revenue_contribution=0.0  # Unknown without segment data
                    )
            
            # 2. Estimate domestic entities based on employee count
            employees = general.get('FullTimeEmployees', 0)
            if employees > 50000:
                # Large company - estimate 3+ domestic entities
                subsidiaries[f"{ticker}_US_Operations"] = SubsidiaryEntity(
                    name=f"{ticker} US Operations",
                    jurisdiction='United States',
                    ownership_percentage=100.0,
                    business_segment='Core Operations',
                    revenue_contribution=60.0
                )
                subsidiaries[f"{ticker}_US_Services"] = SubsidiaryEntity(
                    name=f"{ticker} Services",
                    jurisdiction='United States',
                    ownership_percentage=100.0,
                    business_segment='Services',
                    revenue_contribution=25.0
                )
                subsidiaries[f"{ticker}_US_Technology"] = SubsidiaryEntity(
                    name=f"{ticker} Technology",
                    jurisdiction='United States',
                    ownership_percentage=100.0,
                    business_segment='Technology',
                    revenue_contribution=15.0
                )
            elif employees > 10000:
                # Medium company - estimate 2 domestic entities
                subsidiaries[f"{ticker}_US_Main"] = SubsidiaryEntity(
                    name=f"{ticker} Main Operations",
                    jurisdiction='United States',
                    ownership_percentage=100.0,
                    business_segment='Operations',
                    revenue_contribution=75.0
                )
                subsidiaries[f"{ticker}_US_Support"] = SubsidiaryEntity(
                    name=f"{ticker} Support Services",
                    jurisdiction='United States',
                    ownership_percentage=100.0,
                    business_segment='Services',
                    revenue_contribution=25.0
                )
            elif employees > 1000:
                # Small company - 1 main entity
                subsidiaries[f"{ticker}_US"] = SubsidiaryEntity(
                    name=f"{ticker} Operations",
                    jurisdiction='United States',
                    ownership_percentage=100.0,
                    business_segment=general.get('Sector', 'Operations'),
                    revenue_contribution=100.0
                )
            
            count = len(subsidiaries)
            logger.info(f"📊 Estimated {count} subsidiaries from EODHD data "
                       f"({len(listings)} international listings, {employees:,} employees)")
            
            return subsidiaries
            
        except Exception as e:
            logger.warning(f"Error parsing EODHD listings: {e}")
            return {}

    def _parse_exhibit_21(
            self, filing: Dict[str, Any]) -> Dict[str, SubsidiaryEntity]:
        """Parse Exhibit 21 subsidiary data from SEC filing - DISABLED (use EODHD)"""
        logger.warning("Exhibit 21 parsing disabled - subsidiary data derived from EODHD")
        return {}
    
    def _build_hierarchy(
            self, entities: Dict[str, SubsidiaryEntity]
    ) -> Dict[str, List[str]]:
        """Build corporate hierarchy from entity relationships"""
        hierarchy = defaultdict(list)

        for entity_id, entity in entities.items():
            parent = entity.parent_entity or "ROOT"
            hierarchy[parent].append(entity_id)

        return dict(hierarchy)

    def _calculate_risk_weights(
            self, entities: Dict[str, SubsidiaryEntity]) -> Dict[str, float]:
        """Calculate risk weights based on ownership and revenue"""
        risk_weights = {}

        total_revenue = sum(
            e.revenue_contribution or 0 for e in entities.values())

        for entity_id, entity in entities.items():
            # Base weight from ownership percentage
            ownership_weight = entity.ownership_percentage / 100.0

            # Adjust by revenue contribution
            if entity.revenue_contribution and total_revenue > 0:
                revenue_weight = entity.revenue_contribution / total_revenue
                # Combine ownership and revenue weights
                risk_weight = (ownership_weight * 0.6) + (revenue_weight * 0.4)
            else:
                # Default for unknown revenue
                risk_weight = ownership_weight * 0.5

            risk_weights[entity_id] = risk_weight

        return risk_weights
    
    def _calculate_geographic_exposure(
            self, entities: Dict[str, SubsidiaryEntity]) -> Dict[str, float]:
        """Calculate geographic exposure by jurisdiction"""
        geographic_exposure = defaultdict(float)

        total_weight = sum(
            e.ownership_percentage / 100.0 for e in entities.values())

        for entity in entities.values():
            jurisdiction = self._normalize_jurisdiction(entity.jurisdiction)
            if total_weight > 0:
                weight = (entity.ownership_percentage / 100.0) / total_weight
            else:
                weight = 0
            geographic_exposure[jurisdiction] += weight

        return dict(geographic_exposure)

    def _calculate_segment_exposure(
            self, entities: Dict[str, SubsidiaryEntity]) -> Dict[str, float]:
        """Calculate business segment exposure"""
        segment_exposure = defaultdict(float)

        total_revenue = sum(
            e.revenue_contribution or 0 for e in entities.values())

        if total_revenue > 0:
            for entity in entities.values():
                if entity.business_segment and entity.revenue_contribution:
                    segment = entity.business_segment
                    weight = entity.revenue_contribution / total_revenue
                    segment_exposure[segment] += weight
        else:
            # Equal weight if no revenue data
            segments = {e.business_segment for e in entities.values()
                        if e.business_segment}
            if segments:
                equal_weight = 1.0 / len(segments)
                for segment in segments:
                    segment_exposure[segment] = equal_weight

        return dict(segment_exposure)
    
    def _normalize_jurisdiction(self, jurisdiction: str) -> str:
        """Normalize jurisdiction names for consistency"""
        jurisdiction = jurisdiction.strip().title()
        
        # Common mappings
        mappings = {
            'Delaware': 'US-Delaware',
            'California': 'US-California',
            'New York': 'US-New York',
            'Nevada': 'US-Nevada',
            'United States': UNITED_STATES,
            'Usa': UNITED_STATES,
            'Us': UNITED_STATES,
            'Cayman Islands': CAYMAN_ISLANDS,
            'British Virgin Islands': BRITISH_VIRGIN_ISLANDS,
            'Ireland': 'Ireland',
            'Luxembourg': 'Luxembourg'
        }
        
        return mappings.get(jurisdiction, jurisdiction)
    
    def get_risk_adjusted_metrics(self, structure: CorporateStructure, 
                                 base_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust financial metrics based on subsidiary risk weights
        
        Args:
            structure: Corporate structure with risk weights
            base_metrics: Base financial metrics (revenue, ebitda, etc.)
            
        Returns:
            Risk-adjusted metrics accounting for subsidiary exposures
        """
        adjusted_metrics = base_metrics.copy()
        
        try:
            # Geographic risk adjustments
            if structure.geographic_exposure:
                risk_multiplier = self._calculate_geographic_risk_multiplier(
                    structure.geographic_exposure)
                
                # Apply to valuation-sensitive metrics
                for metric in ['revenue_growth', 'ebitda_margin', 'wacc']:
                    if metric in adjusted_metrics:
                        adjusted_metrics[f'{metric}_geographic_adj'] = (
                            adjusted_metrics[metric] * risk_multiplier)
            
            # Segment concentration risk
            if structure.segment_exposure:
                concentration_risk = self._calculate_concentration_risk(
                    structure.segment_exposure)
                
                # Adjust discount rate for concentration
                if 'wacc' in adjusted_metrics:
                    adjusted_metrics['wacc_concentration_adj'] = (
                        adjusted_metrics['wacc'] + concentration_risk)
            
            # Subsidiary control risk (for minority stakes)
            control_risk = self._calculate_control_risk(structure.entities)
            if 'wacc' in adjusted_metrics:
                adjusted_metrics['wacc_control_adj'] = (
                    adjusted_metrics['wacc'] + control_risk)
        
        except Exception as e:
            logger.warning(f"Error calculating risk adjustments: {e}")
        
        # Add summary risk metrics expected by the calling code
        if structure.geographic_exposure:
            adjusted_metrics['geographic_risk'] = self._calculate_geographic_risk_multiplier(
                structure.geographic_exposure) - 1.0  # Risk premium over base
        else:
            adjusted_metrics['geographic_risk'] = 0.0
            
        if structure.segment_exposure:
            adjusted_metrics['concentration_risk'] = self._calculate_concentration_risk(
                structure.segment_exposure)
        else:
            adjusted_metrics['concentration_risk'] = 0.0
            
        adjusted_metrics['control_risk'] = self._calculate_control_risk(structure.entities)
        
        # Calculate total risk adjustment as weighted average
        adjusted_metrics['total_risk_adjustment'] = (
            adjusted_metrics['geographic_risk'] * 0.4 +
            adjusted_metrics['concentration_risk'] * 0.4 +
            adjusted_metrics['control_risk'] * 0.2
        )
        
        return adjusted_metrics
    
    def _calculate_geographic_risk_multiplier(self, 
                                           geographic_exposure: Dict[str, float]) -> float:
        """Calculate risk multiplier based on geographic exposure"""
        # Country risk scores (simplified)
        country_risk_scores = {
            UNITED_STATES: 1.0,
            'US-Delaware': 1.0,
            'US-California': 1.0,
            'US-New York': 1.0,
            'Ireland': 1.05,
            'Luxembourg': 1.05,
            CAYMAN_ISLANDS: 1.1,
            BRITISH_VIRGIN_ISLANDS: 1.1,
        }
        
        weighted_risk = 0.0
        for country, exposure in geographic_exposure.items():
            risk_score = country_risk_scores.get(country, 1.15)  # Default higher risk
            weighted_risk += exposure * risk_score
        
        return weighted_risk
    
    def _calculate_concentration_risk(self, segment_exposure: Dict[str, float]) -> float:
        """Calculate concentration risk premium"""
        if not segment_exposure:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index for concentration
        hhi = sum(exposure ** 2 for exposure in segment_exposure.values())
        
        # Convert to risk premium (basis points)
        # HHI of 1.0 (complete concentration) adds 100bp risk
        # HHI of 0.2 (diversified) adds minimal risk
        concentration_premium = max(0, (hhi - 0.2) * 0.125)  # 12.5bp per 0.1 HHI
        
        return concentration_premium
    
    def _calculate_control_risk(self, entities: Dict[str, SubsidiaryEntity]) -> float:
        """Calculate control risk premium for minority stakes"""
        if not entities:
            return 0.0
        
        # Find entities with less than 100% ownership
        minority_stakes = [e for e in entities.values() 
                          if e.ownership_percentage < 100.0]
        
        if not minority_stakes:
            return 0.0
        
        # Calculate weighted control risk
        total_revenue = sum(e.revenue_contribution or 0 for e in entities.values())
        control_risk = 0.0
        
        for entity in minority_stakes:
            if entity.revenue_contribution and total_revenue > 0:
                revenue_weight = entity.revenue_contribution / total_revenue
                control_discount = (100 - entity.ownership_percentage) / 100
                # Add risk premium for lack of control
                entity_risk = revenue_weight * control_discount * 0.05  # 5% max premium
                control_risk += entity_risk
        
        return control_risk
    
    def get_subsidiary_features(self, ticker: str) -> Dict[str, float]:
        """
        Get 25 subsidiary features for AI forecasting pipeline
        
        🔵 1. Count Features (4):
           - subsidiary_count: Total number of subsidiaries
           - subsidiary_country_count: Number of unique countries
           - subsidiary_sector_count: Number of unique business segments
           - subsidiary_region_count: Number of unique macro regions
        
        🟧 2. Complexity/Structure (6):
           - subsidiary_complexity_score: log(1 + count)
           - subsidiary_foreign_exposure_pct: % subsidiaries in foreign countries
           - subsidiary_domestic_exposure_pct: % subsidiaries in domestic country
           - subsidiary_hhi_score: Herfindahl-Hirschman Index (concentration)
           - subsidiary_depth_score: subsidiaries per country (complexity)
           - subsidiary_global_sprawl_score: countries per subsidiary (dispersion)
        
        🟩 3. Concentration (5):
           - subsidiary_top_country_pct: % in most common country
           - subsidiary_entropy: Geographic diversification
           - subsidiary_top_region_pct: % in most common region
           - subsidiary_top_sector_pct: % in most common sector
        
        🟦 4. Exposure Risk/Macro (6):
           - emerging_market_exposure_pct: % in EM countries
           - developed_market_exposure_pct: % in DM countries
           - subsidiary_fx_risk_score: Currency volatility risk
           - subsidiary_political_risk_score: Geopolitical risk
        
        🟫 5. Operational Risk (2):
           - subsidiary_avg_employees: Average employees (if available)
           - subsidiary_revenue_coverage_pct: % revenue covered by subs
        
        🔒 6. Robustness/Integrity (2):
           - subsidiary_data_completeness_pct: Data quality score
           - subsidiary_missing_regions_flag: Missing region data flag
        
        Data source: EODHD Fundamentals (Listings + employee data)
        """
        try:
            import math
            structure = self.get_corporate_structure(ticker)
            
            if not structure or not structure.entities:
                logger.warning(f"No subsidiary data available for {ticker}")
                return self._empty_features()
            
            entities = list(structure.entities.values())
            total_count = len(entities)
            
            # Domestic country (US for US-listed)
            domestic_country = 'United States'
            
            # Country-to-region mapping
            region_map = self._get_country_to_region_map()
            
            # EM/DM classification
            em_countries = self._get_emerging_market_countries()
            
            # Currency volatility scores (1.0 = stable, higher = more volatile)
            fx_risk_map = self._get_fx_risk_scores()
            
            # Political risk scores (1.0 = stable, higher = more risk)
            political_risk_map = self._get_political_risk_scores()
            
            # =================================================================
            # 🔵 1. COUNT FEATURES (4)
            # =================================================================
            subsidiary_count = float(total_count)
            subsidiary_country_count = float(len(structure.geographic_exposure))
            subsidiary_sector_count = float(len(structure.segment_exposure))
            
            # NEW: subsidiary_region_count
            regions = set()
            for country in structure.geographic_exposure.keys():
                region = region_map.get(country, 'Other')
                regions.add(region)
            subsidiary_region_count = float(len(regions))
            
            # =================================================================
            # 🟧 2. COMPLEXITY/STRUCTURE FEATURES (6)
            # =================================================================
            subsidiary_complexity_score = math.log(1 + total_count)
            
            domestic_count = sum(1 for e in entities if e.jurisdiction == domestic_country)
            foreign_count = total_count - domestic_count
            subsidiary_domestic_exposure_pct = (domestic_count / total_count) * 100.0
            subsidiary_foreign_exposure_pct = (foreign_count / total_count) * 100.0
            
            # NEW: subsidiary_hhi_score (Herfindahl-Hirschman Index)
            # More precise than entropy for concentration measurement
            subsidiary_hhi_score = 0.0
            if structure.geographic_exposure:
                subsidiary_hhi_score = sum(w**2 for w in structure.geographic_exposure.values()) * 10000
            
            # NEW: subsidiary_depth_score (subsidiaries per country)
            subsidiary_depth_score = subsidiary_count / max(1, subsidiary_country_count)
            
            # NEW: subsidiary_global_sprawl_score (countries per subsidiary)
            subsidiary_global_sprawl_score = subsidiary_country_count / max(1, subsidiary_count)
            
            # =================================================================
            # 🟩 3. CONCENTRATION FEATURES (4)
            # =================================================================
            # Top country percentage
            if structure.geographic_exposure:
                max_country_weight = max(structure.geographic_exposure.values())
                subsidiary_top_country_pct = max_country_weight * 100.0
            else:
                subsidiary_top_country_pct = 100.0 if total_count > 0 else 0.0
            
            # Entropy (Shannon entropy)
            subsidiary_entropy = 0.0
            if structure.geographic_exposure:
                for weight in structure.geographic_exposure.values():
                    if weight > 0:
                        subsidiary_entropy -= weight * math.log(weight)
            
            # NEW: subsidiary_top_region_pct
            region_weights = {}
            for country, weight in structure.geographic_exposure.items():
                region = region_map.get(country, 'Other')
                region_weights[region] = region_weights.get(region, 0.0) + weight
            subsidiary_top_region_pct = max(region_weights.values()) * 100.0 if region_weights else 0.0
            
            # NEW: subsidiary_top_sector_pct
            if structure.segment_exposure:
                subsidiary_top_sector_pct = max(structure.segment_exposure.values()) * 100.0
            else:
                subsidiary_top_sector_pct = 100.0 if total_count > 0 else 0.0
            
            # =================================================================
            # 🟦 4. EXPOSURE RISK / MACRO SENSITIVITY (4)
            # =================================================================
            # NEW: emerging_market_exposure_pct
            em_count = sum(1 for e in entities if e.jurisdiction in em_countries)
            emerging_market_exposure_pct = (em_count / total_count) * 100.0
            
            # NEW: developed_market_exposure_pct
            developed_market_exposure_pct = 100.0 - emerging_market_exposure_pct
            
            # NEW: subsidiary_fx_risk_score (weighted by currency volatility)
            subsidiary_fx_risk_score = 0.0
            for country, weight in structure.geographic_exposure.items():
                fx_risk = fx_risk_map.get(country, 1.5)  # Default medium-high risk
                subsidiary_fx_risk_score += weight * fx_risk
            
            # NEW: subsidiary_political_risk_score (weighted by political stability)
            subsidiary_political_risk_score = 0.0
            for country, weight in structure.geographic_exposure.items():
                pol_risk = political_risk_map.get(country, 2.0)  # Default medium risk
                subsidiary_political_risk_score += weight * pol_risk
            
            # =================================================================
            # 🟫 5. OPERATIONAL RISK FEATURES (2)
            # =================================================================
            # NEW: subsidiary_avg_employees (if available, else 0)
            total_employees = 0
            entities_with_employee_data = 0
            for entity in entities:
                if hasattr(entity, 'employees') and entity.employees:
                    total_employees += entity.employees
                    entities_with_employee_data += 1
            subsidiary_avg_employees = total_employees / max(1, entities_with_employee_data) if entities_with_employee_data > 0 else 0.0
            
            # NEW: subsidiary_revenue_coverage_pct (if revenue data available)
            total_revenue_contribution = sum(e.revenue_contribution or 0 for e in entities)
            subsidiary_revenue_coverage_pct = min(100.0, total_revenue_contribution)
            
            # =================================================================
            # 🔒 6. ROBUSTNESS / INTEGRITY FEATURES (2)
            # =================================================================
            # NEW: subsidiary_data_completeness_pct
            # Check how many entities have complete data
            complete_count = 0
            for entity in entities:
                has_jurisdiction = bool(entity.jurisdiction)
                has_segment = bool(entity.business_segment)
                has_ownership = entity.ownership_percentage > 0
                if has_jurisdiction and has_segment and has_ownership:
                    complete_count += 1
            subsidiary_data_completeness_pct = (complete_count / total_count) * 100.0
            
            # NEW: subsidiary_missing_regions_flag
            # Flag if any entity is in 'Other' region (missing proper region mapping)
            missing_regions = sum(1 for country in structure.geographic_exposure.keys() 
                                if region_map.get(country, 'Other') == 'Other')
            subsidiary_missing_regions_flag = 1.0 if missing_regions > 0 else 0.0
            
            # =================================================================
            # BUILD FEATURE DICT (25 features)
            # =================================================================
            features = {
                # 🔵 1. Count Features (4)
                'subsidiary_count': subsidiary_count,
                'subsidiary_country_count': subsidiary_country_count,
                'subsidiary_sector_count': subsidiary_sector_count,
                'subsidiary_region_count': subsidiary_region_count,
                
                # 🟧 2. Complexity/Structure (6)
                'subsidiary_complexity_score': subsidiary_complexity_score,
                'subsidiary_foreign_exposure_pct': subsidiary_foreign_exposure_pct,
                'subsidiary_domestic_exposure_pct': subsidiary_domestic_exposure_pct,
                'subsidiary_hhi_score': subsidiary_hhi_score,
                'subsidiary_depth_score': subsidiary_depth_score,
                'subsidiary_global_sprawl_score': subsidiary_global_sprawl_score,
                
                # 🟩 3. Concentration (4)
                'subsidiary_top_country_pct': subsidiary_top_country_pct,
                'subsidiary_entropy': subsidiary_entropy,
                'subsidiary_top_region_pct': subsidiary_top_region_pct,
                'subsidiary_top_sector_pct': subsidiary_top_sector_pct,
                
                # 🟦 4. Exposure Risk/Macro (4)
                'emerging_market_exposure_pct': emerging_market_exposure_pct,
                'developed_market_exposure_pct': developed_market_exposure_pct,
                'subsidiary_fx_risk_score': subsidiary_fx_risk_score,
                'subsidiary_political_risk_score': subsidiary_political_risk_score,
                
                # 🟫 5. Operational Risk (2)
                'subsidiary_avg_employees': subsidiary_avg_employees,
                'subsidiary_revenue_coverage_pct': subsidiary_revenue_coverage_pct,
                
                # 🔒 6. Robustness/Integrity (2)
                'subsidiary_data_completeness_pct': subsidiary_data_completeness_pct,
                'subsidiary_missing_regions_flag': subsidiary_missing_regions_flag,
            }
            
            logger.info(
                f"📊 Generated {len(features)} subsidiary features for {ticker}: "
                f"{total_count} subs across {subsidiary_country_count:.0f} countries, "
                f"{subsidiary_region_count:.0f} regions"
            )
            return features
            
        except Exception as e:
            logger.warning(f"Failed to generate subsidiary features for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_features()
    
    def _empty_features(self) -> Dict[str, float]:
        """Return zero values for all 25 subsidiary features"""
        return {
            # 🔵 1. Count Features (4)
            'subsidiary_count': 0.0,
            'subsidiary_country_count': 0.0,
            'subsidiary_sector_count': 0.0,
            'subsidiary_region_count': 0.0,
            
            # 🟧 2. Complexity/Structure (6)
            'subsidiary_complexity_score': 0.0,
            'subsidiary_foreign_exposure_pct': 0.0,
            'subsidiary_domestic_exposure_pct': 0.0,
            'subsidiary_hhi_score': 0.0,
            'subsidiary_depth_score': 0.0,
            'subsidiary_global_sprawl_score': 0.0,
            
            # 🟩 3. Concentration (4)
            'subsidiary_top_country_pct': 0.0,
            'subsidiary_entropy': 0.0,
            'subsidiary_top_region_pct': 0.0,
            'subsidiary_top_sector_pct': 0.0,
            
            # 🟦 4. Exposure Risk/Macro (4)
            'emerging_market_exposure_pct': 0.0,
            'developed_market_exposure_pct': 0.0,
            'subsidiary_fx_risk_score': 0.0,
            'subsidiary_political_risk_score': 0.0,
            
            # 🟫 5. Operational Risk (2)
            'subsidiary_avg_employees': 0.0,
            'subsidiary_revenue_coverage_pct': 0.0,
            
            # 🔒 6. Robustness/Integrity (2)
            'subsidiary_data_completeness_pct': 0.0,
            'subsidiary_missing_regions_flag': 0.0,
        }
    
    def _get_country_to_region_map(self) -> Dict[str, str]:
        """Map countries to macro regions (North America, Europe, APAC, LATAM, Middle East, Africa)"""
        return {
            # North America
            'United States': 'North America',
            'Canada': 'North America',
            'Mexico': 'North America',
            
            # Europe
            'United Kingdom': 'Europe',
            'Germany': 'Europe',
            'France': 'Europe',
            'Ireland': 'Europe',
            'Netherlands': 'Europe',
            'Switzerland': 'Europe',
            'Luxembourg': 'Europe',
            'Belgium': 'Europe',
            'Spain': 'Europe',
            'Italy': 'Europe',
            'Sweden': 'Europe',
            'Norway': 'Europe',
            'Denmark': 'Europe',
            'Finland': 'Europe',
            'Poland': 'Europe',
            'Austria': 'Europe',
            'Portugal': 'Europe',
            'Greece': 'Europe',
            
            # APAC (Asia-Pacific)
            'China': 'APAC',
            'Japan': 'APAC',
            'South Korea': 'APAC',
            'India': 'APAC',
            'Singapore': 'APAC',
            'Hong Kong': 'APAC',
            'Australia': 'APAC',
            'New Zealand': 'APAC',
            'Taiwan': 'APAC',
            'Thailand': 'APAC',
            'Malaysia': 'APAC',
            'Indonesia': 'APAC',
            'Philippines': 'APAC',
            'Vietnam': 'APAC',
            
            # LATAM (Latin America)
            'Brazil': 'LATAM',
            'Argentina': 'LATAM',
            'Chile': 'LATAM',
            'Colombia': 'LATAM',
            'Peru': 'LATAM',
            'Venezuela': 'LATAM',
            'Ecuador': 'LATAM',
            'Uruguay': 'LATAM',
            'Cayman Islands': 'LATAM',
            'British Virgin Islands': 'LATAM',
            'Bermuda': 'LATAM',
            
            # Middle East
            'Saudi Arabia': 'Middle East',
            'United Arab Emirates': 'Middle East',
            'Israel': 'Middle East',
            'Qatar': 'Middle East',
            'Kuwait': 'Middle East',
            'Bahrain': 'Middle East',
            'Oman': 'Middle East',
            'Turkey': 'Middle East',
            'Jordan': 'Middle East',
            'Lebanon': 'Middle East',
            
            # Africa
            'South Africa': 'Africa',
            'Nigeria': 'Africa',
            'Egypt': 'Africa',
            'Kenya': 'Africa',
            'Morocco': 'Africa',
            'Ghana': 'Africa',
            'Ethiopia': 'Africa',
            'Tanzania': 'Africa',
        }
    
    def _get_emerging_market_countries(self) -> set:
        """List of emerging market countries for risk classification"""
        return {
            # BRICS+
            'Brazil', 'Russia', 'India', 'China', 'South Africa',
            
            # Latin America EM
            'Argentina', 'Chile', 'Colombia', 'Peru', 'Mexico',
            
            # Asia EM
            'Indonesia', 'Malaysia', 'Thailand', 'Philippines', 'Vietnam',
            'Pakistan', 'Bangladesh', 'Taiwan',
            
            # EMEA EM
            'Turkey', 'Poland', 'Egypt', 'Nigeria', 'Kenya',
            'Morocco', 'Saudi Arabia', 'United Arab Emirates', 'Qatar',
        }
    
    def _get_fx_risk_scores(self) -> Dict[str, float]:
        """
        FX risk scores: 1.0 = stable (USD, EUR), higher = more volatile
        Used for subsidiary_fx_risk_score calculation
        """
        return {
            # Very stable (reserve currencies)
            'United States': 1.0,
            'Germany': 1.0,
            'France': 1.0,
            'Netherlands': 1.0,
            'Belgium': 1.0,
            'Switzerland': 1.05,
            'Japan': 1.1,
            'United Kingdom': 1.15,
            
            # Stable developed
            'Canada': 1.2,
            'Australia': 1.2,
            'Sweden': 1.2,
            'Norway': 1.2,
            'Denmark': 1.15,
            'Singapore': 1.1,
            'Hong Kong': 1.1,
            
            # Moderate risk
            'South Korea': 1.3,
            'Taiwan': 1.3,
            'Israel': 1.3,
            'Poland': 1.4,
            
            # Higher risk (EM currencies)
            'Brazil': 1.8,
            'Mexico': 1.6,
            'India': 1.5,
            'China': 1.4,
            'Indonesia': 1.7,
            'Thailand': 1.5,
            'Malaysia': 1.5,
            'Turkey': 2.0,
            'South Africa': 1.8,
            'Russia': 2.5,
            'Argentina': 3.0,
        }
    
    def _get_political_risk_scores(self) -> Dict[str, float]:
        """
        Political risk scores: 1.0 = very stable, higher = more risk
        Used for subsidiary_political_risk_score calculation
        """
        return {
            # Very stable (lowest political risk)
            'Switzerland': 1.0,
            'Norway': 1.0,
            'Sweden': 1.0,
            'Denmark': 1.0,
            'Canada': 1.0,
            'Australia': 1.0,
            'New Zealand': 1.0,
            'Netherlands': 1.0,
            'Germany': 1.05,
            'United States': 1.1,
            'United Kingdom': 1.1,
            'Japan': 1.1,
            'Singapore': 1.1,
            
            # Stable
            'France': 1.2,
            'Ireland': 1.1,
            'Belgium': 1.2,
            'Austria': 1.1,
            'Finland': 1.0,
            'South Korea': 1.3,
            'Taiwan': 1.3,
            'Israel': 1.4,
            
            # Moderate risk
            'Spain': 1.3,
            'Italy': 1.4,
            'Poland': 1.4,
            'Czech Republic': 1.3,
            'Chile': 1.4,
            'Uruguay': 1.3,
            'Hong Kong': 1.5,
            
            # Elevated risk
            'Brazil': 1.6,
            'Mexico': 1.6,
            'India': 1.5,
            'China': 1.7,
            'Indonesia': 1.6,
            'Thailand': 1.5,
            'Malaysia': 1.4,
            'South Africa': 1.7,
            'Colombia': 1.6,
            'Peru': 1.5,
            
            # High risk
            'Turkey': 2.0,
            'Argentina': 2.2,
            'Russia': 2.5,
            'Venezuela': 3.0,
            'Egypt': 1.8,
            'Nigeria': 1.9,
            'Pakistan': 2.0,
        }

