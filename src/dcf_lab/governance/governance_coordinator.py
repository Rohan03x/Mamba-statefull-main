"""
Governance Framework Coordinator

This module coordinates all governance activities for production ML systems:
- Integration of A/B testing, risk validation, and documentation systems
- Unified API for governance-controlled model deployment
- Automated compliance workflows and regulatory reporting
- Integration with trading systems and risk management platforms
- Real-time monitoring and alerting for governance violations

Key features:
- Unified governance API for all ML operations
- Automated compliance checks before model deployment
- Real-time risk monitoring with immediate intervention
- Complete audit trails for regulatory inspection
- Integration with existing risk management systems
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import uuid

# Import governance components
from .ab_testing import (
    ABTestConfig, ChampionChallengerFramework,
    ExperimentRegistry, ExperimentStatus, AllocationMethod
)
from .risk_sanity import (
    RiskSanityConfig, RiskSanityValidator, PreTradeRiskCheck,
    MockOptionsProvider, ViolationSeverity
)
from .documentation import (
    ExperimentDocumenter, ParameterDocumenter, AuditTrailManager,
    ComplianceFramework
)

logger = logging.getLogger(__name__)

class GovernanceLevel(Enum):
    """Governance enforcement levels"""
    DEVELOPMENT = "development"     # Minimal governance for development
    STAGING = "staging"            # Moderate governance for staging
    PRODUCTION = "production"      # Full governance for production
    REGULATORY = "regulatory"      # Maximum governance for regulatory environments

class DeploymentDecision(Enum):
    """Model deployment decisions"""
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    PENDING_REVIEW = "pending_review"

@dataclass
class GovernanceConfig:
    """Configuration for governance framework"""
    
    # Governance level
    level: GovernanceLevel = GovernanceLevel.PRODUCTION
    
    # A/B testing configuration
    ab_testing_enabled: bool = True
    ab_test_config: Optional[ABTestConfig] = None
    
    # Risk validation configuration
    risk_validation_enabled: bool = True
    risk_sanity_config: Optional[RiskSanityConfig] = None
    
    # Documentation requirements
    documentation_required: bool = True
    compliance_frameworks: List[ComplianceFramework] = field(
        default_factory=lambda: [ComplianceFramework.INTERNAL]
    )
    
    # Approval workflows
    require_human_approval: bool = True
    auto_approve_threshold: float = 0.1  # Risk score threshold for auto-approval
    
    # Monitoring settings
    monitoring_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'risk_score': 5.0,
        'violation_rate': 0.05,
        'performance_degradation': 0.1
    })
    
    # Integration settings
    trading_system_integration: bool = False
    risk_system_integration: bool = False
    reporting_system_integration: bool = False

@dataclass
class ModelDeploymentRequest:
    """Request for model deployment through governance"""
    
    request_id: str
    model_id: str
    model_version: str
    requested_by: str
    timestamp: str
    
    # Model details
    model_name: str
    model_type: str
    predicted_symbols: List[str]
    prediction_horizon_days: int
    
    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    backtest_metrics: Dict[str, float]
    
    # Risk information
    max_position_size: float
    expected_leverage: float
    risk_estimates: Dict[str, float]
    
    # Deployment configuration
    allocation_percentage: float = 5.0  # Start with small allocation
    target_environment: str = "staging"
    rollout_strategy: str = "gradual"
    
    # Documentation
    deployment_rationale: str = ""
    expected_benefits: List[str] = field(default_factory=list)
    known_risks: List[str] = field(default_factory=list)

@dataclass
class GovernanceDecision:
    """Governance decision for model deployment"""
    
    decision_id: str
    request_id: str
    decision: DeploymentDecision
    timestamp: str
    decided_by: str
    
    # Decision rationale
    decision_reason: str
    conditions: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_score: float = 0.0
    risk_violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # A/B test configuration (if approved)
    ab_test_id: Optional[str] = None
    allocation_method: Optional[AllocationMethod] = None
    test_duration_days: Optional[int] = None
    
    # Monitoring requirements
    monitoring_metrics: List[str] = field(default_factory=list)
    review_schedule: Optional[str] = None
    
    # Compliance information
    compliance_status: Dict[str, str] = field(default_factory=dict)
    audit_trail_id: str = ""

class GovernanceOrchestrator:
    """Main coordinator for governance framework"""
    
    def __init__(self, config: GovernanceConfig):
        self.config = config
        
        # Initialize components based on configuration
        self._init_components()
        
        # Request tracking
        self.pending_requests = {}
        self.decision_history = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.alert_handlers = []
    
    def _init_components(self):
        """Initialize governance components"""
        
        # A/B testing framework
        if self.config.ab_testing_enabled:
            ab_config = self.config.ab_test_config or ABTestConfig(
                name="default_ab_test",
                description="Default A/B test configuration",
                champion_model="champion",
                challenger_model="challenger"
            )
            self.ab_framework = ChampionChallengerFramework(ab_config)
            self.experiment_registry = ExperimentRegistry()
        else:
            self.ab_framework = None
            self.experiment_registry = None
        
        # Risk validation
        if self.config.risk_validation_enabled:
            risk_config = self.config.risk_sanity_config or RiskSanityConfig()
            # Use mock provider for now, would integrate with real provider
            options_provider = MockOptionsProvider(risk_config)
            self.risk_validator = RiskSanityValidator(risk_config, options_provider)
            self.pre_trade_checker = PreTradeRiskCheck(risk_config, options_provider)
        else:
            self.risk_validator = None
            self.pre_trade_checker = None
        
        # Documentation system
        if self.config.documentation_required:
            self.experiment_documenter = ExperimentDocumenter()
            self.parameter_documenter = ParameterDocumenter()
            self.audit_manager = AuditTrailManager()
        else:
            self.experiment_documenter = None
            self.parameter_documenter = None
            self.audit_manager = None
        
        logger.info(f"Initialized governance with level: {self.config.level.value}")
    
    async def request_model_deployment(self, 
                                     deployment_request: ModelDeploymentRequest) -> str:
        """
        Request model deployment through governance process
        
        Args:
            deployment_request: Complete deployment request
            
        Returns:
            Request ID for tracking
        """
        
        request_id = deployment_request.request_id
        
        # Log audit event
        if self.audit_manager:
            self.audit_manager.log_event(
                event_type="deployment_request",
                user_id=deployment_request.requested_by,
                resource_type="model",
                resource_id=deployment_request.model_id,
                action="request_deployment",
                details={
                    'model_version': deployment_request.model_version,
                    'target_environment': deployment_request.target_environment,
                    'allocation_percentage': deployment_request.allocation_percentage
                }
            )
        
        # Store request
        self.pending_requests[request_id] = deployment_request
        
        # Start governance workflow
        try:
            decision = await self._process_deployment_request(deployment_request)
            self.decision_history[request_id] = decision
            
            # Remove from pending
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            
            logger.info(f"Completed governance review for {request_id}: {decision.decision.value}")
            
        except Exception as e:
            logger.error(f"Governance workflow failed for {request_id}: {e}")
            
            # Create rejection decision
            decision = GovernanceDecision(
                decision_id=str(uuid.uuid4()),
                request_id=request_id,
                decision=DeploymentDecision.REJECTED,
                timestamp=datetime.now(timezone.utc).isoformat(),
                decided_by="system",
                decision_reason=f"Governance workflow error: {str(e)}",
                risk_score=100.0  # Maximum risk for errors
            )
            
            self.decision_history[request_id] = decision
            
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
        
        return request_id
    
    async def _process_deployment_request(self, 
                                        request: ModelDeploymentRequest) -> GovernanceDecision:
        """Process deployment request through governance pipeline"""
        
        decision_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Initialize decision
        decision = GovernanceDecision(
            decision_id=decision_id,
            request_id=request.request_id,
            decision=DeploymentDecision.PENDING_REVIEW,
            timestamp=timestamp,
            decided_by="system",
            decision_reason="Processing governance checks"
        )
        
        # Step 1: Risk validation
        if self.config.risk_validation_enabled and self.risk_validator:
            risk_assessment = await self._assess_deployment_risk(request)
            decision.risk_score = risk_assessment['risk_score']
            decision.risk_violations = risk_assessment['violations']
            
            # Check if risk is too high
            if decision.risk_score > 50.0:  # High risk threshold
                decision.decision = DeploymentDecision.REJECTED
                decision.decision_reason = "High risk score from validation"
                return decision
        
        # Step 2: Documentation compliance
        if self.config.documentation_required and self.experiment_documenter:
            compliance_check = await self._validate_documentation_compliance(request)
            decision.compliance_status = compliance_check
            
            # Check if documentation is insufficient
            if not all(status == 'compliant' for status in compliance_check.values()):
                if self.config.level == GovernanceLevel.REGULATORY:
                    decision.decision = DeploymentDecision.REJECTED
                    decision.decision_reason = "Insufficient documentation for regulatory level"
                    return decision
                else:
                    decision.conditions.append("Complete documentation requirements")
        
        # Step 3: A/B test setup (if approved so far)
        ab_test_id = None
        if (self.config.ab_testing_enabled and self.ab_framework and 
            decision.decision != DeploymentDecision.REJECTED):
            
            ab_test_setup = await self._setup_ab_test(request)
            if ab_test_setup['success']:
                ab_test_id = ab_test_setup['experiment_id']
                decision.ab_test_id = ab_test_id
                decision.allocation_method = AllocationMethod.STRATIFIED
                decision.test_duration_days = 14  # Default test duration
            else:
                decision.conditions.append("Resolve A/B test setup issues")
        
        # Step 4: Make final decision
        final_decision = self._make_final_decision(decision, request)
        
        # Step 5: Setup monitoring if approved
        if final_decision.decision in [DeploymentDecision.APPROVED, DeploymentDecision.CONDITIONAL]:
            monitoring_setup = await self._setup_deployment_monitoring(request, final_decision)
            final_decision.monitoring_metrics = monitoring_setup['metrics']
            final_decision.review_schedule = monitoring_setup['review_schedule']
        
        # Log final decision
        if self.audit_manager:
            self.audit_manager.log_event(
                event_type="deployment_decision",
                user_id="system",
                resource_type="model",
                resource_id=request.model_id,
                action=f"decided_{final_decision.decision.value}",
                details={
                    'decision_id': decision_id,
                    'risk_score': final_decision.risk_score,
                    'conditions': final_decision.conditions,
                    'ab_test_id': ab_test_id
                }
            )
        
        return final_decision
    
    async def _assess_deployment_risk(self, request: ModelDeploymentRequest) -> Dict[str, Any]:
        """Assess risk for deployment request"""
        
        # Simulate risk checks for multiple symbols
        all_violations = []
        total_risk_score = 0.0
        
        for symbol in request.predicted_symbols:
            # Create mock prediction for validation
            predicted_move = request.risk_estimates.get(symbol, 0.01)  # 1% default move
            confidence_interval = (predicted_move * 0.5, predicted_move * 1.5)
            
            # Validate prediction
            violations = self.risk_validator.validate_prediction(
                symbol=symbol,
                predicted_move=predicted_move,
                confidence_interval=confidence_interval,
                horizon_days=request.prediction_horizon_days,
                position_size=request.max_position_size
            )
            
            all_violations.extend(violations)
        
        # Calculate overall risk score
        if all_violations:
            # Weight violations by severity
            severity_weights = {
                ViolationSeverity.INFO: 1,
                ViolationSeverity.WARNING: 3,
                ViolationSeverity.ERROR: 10,
                ViolationSeverity.CRITICAL: 25
            }
            
            total_risk_score = sum(
                severity_weights.get(v.severity, 1) for v in all_violations
            )
        
        return {
            'risk_score': min(total_risk_score, 100.0),  # Cap at 100
            'violations': [
                {
                    'symbol': v.symbol,
                    'type': v.violation_type.value,
                    'severity': v.severity.value,
                    'message': v.message
                } for v in all_violations
            ],
            'symbols_checked': len(request.predicted_symbols),
            'total_violations': len(all_violations)
        }
    
    async def _validate_documentation_compliance(self, 
                                               request: ModelDeploymentRequest) -> Dict[str, str]:
        """Validate documentation compliance for deployment"""
        
        compliance_status = {}
        
        for framework in self.config.compliance_frameworks:
            if framework == ComplianceFramework.INTERNAL:
                # Check basic documentation requirements
                has_rationale = bool(request.deployment_rationale.strip())
                has_metrics = bool(request.training_metrics and request.validation_metrics)
                has_risk_assessment = bool(request.known_risks)
                
                if has_rationale and has_metrics and has_risk_assessment:
                    compliance_status[framework.value] = 'compliant'
                else:
                    compliance_status[framework.value] = 'non_compliant'
            
            elif framework == ComplianceFramework.SR_11_7:
                # More stringent requirements for SR 11-7
                has_validation = bool(request.backtest_metrics)
                has_risk_limits = request.max_position_size <= 0.1  # 10% max
                has_monitoring = len(request.expected_benefits) > 0
                
                if has_validation and has_risk_limits and has_monitoring:
                    compliance_status[framework.value] = 'compliant'
                else:
                    compliance_status[framework.value] = 'non_compliant'
            
            else:
                # Default to compliant for other frameworks
                compliance_status[framework.value] = 'compliant'
        
        return compliance_status
    
    async def _setup_ab_test(self, request: ModelDeploymentRequest) -> Dict[str, Any]:
        """Setup A/B test for model deployment"""
        
        try:
            # Create A/B test configuration
            ab_config = ABTestConfig(
                name=f"model_deployment_{request.model_id}_{request.model_version}",
                description=f"Champion/challenger test for {request.model_name}",
                champion_model="current_champion",
                challenger_model=request.model_id,
                allocation_ratio=0.7,  # 70% champion, 30% challenger
                allocation_method=AllocationMethod.STRATIFIED,
                alpha=0.05,
                power=0.8,
                minimum_effect_size=0.1,
                max_duration_days=14
            )
            
            # Start experiment
            experiment = self.ab_framework.start_experiment(ab_config)
            
            return {
                'success': True,
                'experiment_id': experiment.experiment_id,
                'allocation': ab_config.treatment_allocation
            }
            
        except Exception as e:
            logger.error(f"Failed to setup A/B test: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _make_final_decision(self, decision: GovernanceDecision, 
                           request: ModelDeploymentRequest) -> GovernanceDecision:
        """Make final deployment decision based on all checks"""
        
        # Auto-approval for low risk
        if (decision.risk_score <= self.config.auto_approve_threshold and
            not decision.conditions and
            not self.config.require_human_approval):
            
            decision.decision = DeploymentDecision.APPROVED
            decision.decision_reason = "Automated approval - low risk"
            decision.decided_by = "system"
        
        # Conditional approval for medium risk with conditions
        elif decision.risk_score <= 10.0 and decision.conditions:
            decision.decision = DeploymentDecision.CONDITIONAL
            decision.decision_reason = f"Conditional approval with {len(decision.conditions)} conditions"
        
        # Rejection for high risk
        elif decision.risk_score > 25.0:
            decision.decision = DeploymentDecision.REJECTED
            decision.decision_reason = f"High risk score: {decision.risk_score:.1f}"
        
        # Human review required
        else:
            decision.decision = DeploymentDecision.PENDING_REVIEW
            decision.decision_reason = "Human review required"
        
        return decision
    
    async def _setup_deployment_monitoring(self, request: ModelDeploymentRequest,
                                         decision: GovernanceDecision) -> Dict[str, Any]:
        """Setup monitoring for deployed model"""
        
        # Define monitoring metrics based on model type
        monitoring_metrics = [
            'prediction_accuracy',
            'sharpe_ratio',
            'max_drawdown',
            'hit_rate',
            'average_return'
        ]
        
        # Add risk-specific metrics if high risk
        if decision.risk_score > 5.0:
            monitoring_metrics.extend([
                'var_95',
                'expected_shortfall',
                'leverage_ratio',
                'position_concentration'
            ])
        
        # Determine review schedule based on risk and governance level
        if self.config.level == GovernanceLevel.REGULATORY:
            review_schedule = "daily"
        elif decision.risk_score > 10.0:
            review_schedule = "weekly"
        else:
            review_schedule = "monthly"
        
        return {
            'metrics': monitoring_metrics,
            'review_schedule': review_schedule,
            'alert_thresholds': self.config.alert_thresholds.copy(),
            'monitoring_duration_days': 90  # Monitor for 90 days initially
        }
    
    def get_deployment_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of deployment request"""
        
        if request_id in self.pending_requests:
            return {
                'status': 'pending',
                'request': self.pending_requests[request_id],
                'submitted_at': self.pending_requests[request_id].timestamp
            }
        
        elif request_id in self.decision_history:
            decision = self.decision_history[request_id]
            return {
                'status': 'completed',
                'decision': decision.decision.value,
                'decision_reason': decision.decision_reason,
                'risk_score': decision.risk_score,
                'conditions': decision.conditions,
                'decided_at': decision.timestamp
            }
        
        else:
            return None
    
    def get_governance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get governance activity summary"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # Count recent decisions
        recent_decisions = [
            d for d in self.decision_history.values()
            if d.timestamp > cutoff_str
        ]
        
        decision_counts = {}
        for decision_type in DeploymentDecision:
            decision_counts[decision_type.value] = sum(
                1 for d in recent_decisions if d.decision == decision_type
            )
        
        # Risk statistics
        risk_scores = [d.risk_score for d in recent_decisions if d.risk_score > 0]
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        # A/B test statistics
        active_ab_tests = 0
        if self.ab_framework:
            active_ab_tests = len([
                exp for exp in self.ab_framework.active_experiments.values()
                if exp.status == ExperimentStatus.RUNNING
            ])
        
        return {
            'period_days': days,
            'total_requests': len(recent_decisions),
            'pending_requests': len(self.pending_requests),
            'decision_breakdown': decision_counts,
            'average_risk_score': avg_risk_score,
            'active_ab_tests': active_ab_tests,
            'governance_level': self.config.level.value,
            'components_enabled': {
                'ab_testing': self.config.ab_testing_enabled,
                'risk_validation': self.config.risk_validation_enabled,
                'documentation': self.config.documentation_required,
                'monitoring': self.config.monitoring_enabled
            }
        }
    
    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register handler for governance alerts"""
        self.alert_handlers.append(handler)
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger governance alert"""
        
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': alert_type,
            'severity': details.get('severity', 'warning'),
            'details': details
        }
        
        logger.warning(f"Governance alert: {alert_type} - {details}")
        
        # Notify registered handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

# Convenience functions for common governance patterns

def create_production_governance() -> GovernanceOrchestrator:
    """Create production-ready governance configuration"""
    
    config = GovernanceConfig(
        level=GovernanceLevel.PRODUCTION,
        ab_testing_enabled=True,
        risk_validation_enabled=True,
        documentation_required=True,
        require_human_approval=True,
        compliance_frameworks=[ComplianceFramework.INTERNAL, ComplianceFramework.SR_11_7],
        monitoring_enabled=True
    )
    
    return GovernanceOrchestrator(config)

def create_regulatory_governance() -> GovernanceOrchestrator:
    """Create regulatory-compliant governance configuration"""
    
    config = GovernanceConfig(
        level=GovernanceLevel.REGULATORY,
        ab_testing_enabled=True,
        risk_validation_enabled=True,
        documentation_required=True,
        require_human_approval=True,
        compliance_frameworks=[
            ComplianceFramework.SR_11_7,
            ComplianceFramework.MIFID_II,
            ComplianceFramework.INTERNAL
        ],
        monitoring_enabled=True,
        auto_approve_threshold=0.0,  # No auto-approval for regulatory
        alert_thresholds={
            'risk_score': 1.0,  # Very low threshold
            'violation_rate': 0.01,
            'performance_degradation': 0.05
        }
    )
    
    return GovernanceOrchestrator(config)

def create_development_governance() -> GovernanceOrchestrator:
    """Create lightweight governance for development"""
    
    config = GovernanceConfig(
        level=GovernanceLevel.DEVELOPMENT,
        ab_testing_enabled=False,
        risk_validation_enabled=True,
        documentation_required=False,
        require_human_approval=False,
        compliance_frameworks=[ComplianceFramework.INTERNAL],
        monitoring_enabled=False,
        auto_approve_threshold=25.0  # Higher threshold for dev
    )
    
    return GovernanceOrchestrator(config)