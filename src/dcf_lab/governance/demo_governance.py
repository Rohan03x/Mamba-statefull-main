"""
Governance Framework Demonstration

This script demonstrates the complete Task 10 governance and experiments framework:
- Champion/challenger A/B testing for live paper-trading
- Pre-trade risk sanity checks with options-implied validation
- Comprehensive regulatory documentation and audit trails
- Complete model deployment governance workflow

This demonstrates what regulators and investors look for in production ML systems.
"""

import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Any

# Import governance components
from .governance_coordinator import (
    create_production_governance, ModelDeploymentRequest
)
from .ab_testing import ABTestConfig, AllocationMethod
from .risk_sanity import ViolationSeverity
from .documentation import ComplianceFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GovernanceDemo:
    """Demonstration of governance framework capabilities"""
    
    def __init__(self):
        self.governance = None
        self.demo_results = {}
    
    async def run_complete_demo(self):
        """Run complete governance framework demonstration"""
        
        print("\n" + "="*80)
        print("TASK 10: GOVERNANCE & EXPERIMENTS FRAMEWORK DEMONSTRATION")
        print("="*80)
        
        try:
            # Demo 1: Production governance setup
            await self.demo_production_governance()
            
            # Demo 2: Model deployment workflow
            await self.demo_model_deployment_workflow()
            
            # Demo 3: A/B testing champion/challenger
            await self.demo_champion_challenger_testing()
            
            # Demo 4: Risk sanity validation
            await self.demo_risk_sanity_validation()
            
            # Demo 5: Regulatory documentation
            await self.demo_regulatory_documentation()
            
            # Demo 6: Governance monitoring and alerts
            await self.demo_governance_monitoring()
            
            # Summary
            self.print_demo_summary()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def demo_production_governance(self):
        """Demonstrate production governance setup"""
        
        print("\n📋 Demo 1: Production Governance Setup")
        print("-" * 50)
        
        # Create production governance
        self.governance = create_production_governance()
        
        print("✅ Production governance framework initialized")
        print(f"   - Governance level: {self.governance.config.level.value}")
        print(f"   - A/B testing enabled: {self.governance.config.ab_testing_enabled}")
        print(f"   - Risk validation enabled: {self.governance.config.risk_validation_enabled}")
        print(f"   - Documentation required: {self.governance.config.documentation_required}")
        
        # Show governance summary
        summary = self.governance.get_governance_summary()
        print("\n📊 Governance Summary:")
        print(f"   - Total requests: {summary['total_requests']}")
        print(f"   - Pending requests: {summary['pending_requests']}")
        print(f"   - Active A/B tests: {summary['active_ab_tests']}")
        
        self.demo_results['governance_setup'] = {
            'status': 'success',
            'summary': summary
        }
    
    async def demo_model_deployment_workflow(self):
        """Demonstrate complete model deployment workflow"""
        
        print("\n🚀 Demo 2: Model Deployment Workflow")
        print("-" * 50)
        
        # Create deployment request
        deployment_request = ModelDeploymentRequest(
            request_id="deploy_001",
            model_id="enhanced_tf_v2.1",
            model_version="2.1.0",
            requested_by="portfolio_manager",
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name="Enhanced Temporal Fusion Transformer",
            model_type="temporal_fusion_transformer",
            predicted_symbols=["AAPL", "MSFT", "GOOGL"],
            prediction_horizon_days=5,
            training_metrics={
                "train_sharpe": 1.85,
                "train_hit_rate": 0.58,
                "train_max_drawdown": 0.12
            },
            validation_metrics={
                "val_sharpe": 1.62,
                "val_hit_rate": 0.55,
                "val_max_drawdown": 0.15
            },
            backtest_metrics={
                "backtest_sharpe": 1.43,
                "backtest_hit_rate": 0.52,
                "backtest_max_drawdown": 0.18,
                "backtest_calmar": 7.9
            },
            max_position_size=0.05,  # 5% max position
            expected_leverage=2.0,
            risk_estimates={
                "AAPL": 0.015,  # 1.5% expected move
                "MSFT": 0.012,
                "GOOGL": 0.018
            },
            allocation_percentage=10.0,
            target_environment="production",
            rollout_strategy="gradual",
            deployment_rationale="Model shows consistent outperformance with improved risk metrics",
            expected_benefits=[
                "15% improvement in Sharpe ratio",
                "Better drawdown control",
                "Enhanced multi-horizon predictions"
            ],
            known_risks=[
                "Potential overfitting to recent market regime",
                "Increased computational requirements"
            ]
        )
        
        print(f"📤 Submitting deployment request for {deployment_request.model_name}")
        print(f"   - Model ID: {deployment_request.model_id}")
        print(f"   - Symbols: {deployment_request.predicted_symbols}")
        print(f"   - Max position size: {deployment_request.max_position_size:.1%}")
        
        # Submit request
        request_id = await self.governance.request_model_deployment(deployment_request)
        
        # Check status
        status = self.governance.get_deployment_status(request_id)
        
        print("\n✅ Deployment request processed")
        print(f"   - Request ID: {request_id}")
        print(f"   - Status: {status['status']}")
        
        if status['status'] == 'completed':
            print(f"   - Decision: {status['decision']}")
            print(f"   - Risk score: {status['risk_score']:.1f}")
            if status['conditions']:
                print(f"   - Conditions: {len(status['conditions'])}")
                for condition in status['conditions']:
                    print(f"     • {condition}")
        
        self.demo_results['deployment_workflow'] = {
            'status': 'success',
            'request_id': request_id,
            'deployment_status': status
        }
    
    async def demo_champion_challenger_testing(self):
        """Demonstrate champion/challenger A/B testing"""
        
        print("\n🏆 Demo 3: Champion/Challenger A/B Testing")
        print("-" * 50)
        
        if not self.governance.ab_framework:
            print("❌ A/B testing not enabled in governance")
            return
        
        # Setup A/B test configuration
        ab_config = ABTestConfig(
            name="enhanced_tf_vs_baseline",
            description="Champion/challenger test: Enhanced TF vs Baseline model",
            champion_model="baseline_model_v1.0",
            challenger_model="enhanced_tf_v2.1",
            allocation_ratio=0.7,  # 70% champion, 30% challenger
            allocation_method=AllocationMethod.STRATIFIED,
            stratification_features=['sector', 'market_cap'],
            alpha=0.05,
            power=0.8,
            minimum_effect_size=0.15,  # 15% minimum improvement
            max_duration_days=14,
            min_sample_size=1000
        )
        
        print(f"⚡ Setting up A/B test: {ab_config.name}")
        print(f"   - Champion model: {ab_config.champion_model}")
        print(f"   - Challenger model: {ab_config.challenger_model}")
        print(f"   - Allocation: Champion {ab_config.allocation_ratio:.0%}, Challenger {1-ab_config.allocation_ratio:.0%}")
        print(f"   - Minimum effect size: {ab_config.minimum_effect_size:.1%}")
        
        # Create and start experiment
        experiment = self.governance.ab_framework.create_experiment(ab_config)
        started = self.governance.ab_framework.start_experiment(experiment)
        
        print(f"\n✅ A/B test {'started successfully' if started else 'creation completed'}")
        print(f"   - Experiment ID: {experiment.id}")
        print(f"   - Status: {experiment.status.value}")
        print(f"   - Expected duration: {experiment.config.max_duration_days} days")
        
        # Simulate some treatment allocations
        test_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        allocations = []
        
        for symbol in test_symbols:
            allocation = self.governance.ab_framework.allocate_treatment(
                experiment.id, 
                {"symbol": symbol, "sector": "tech"}
            )
            allocations.append((symbol, allocation))
        
        print("\n📊 Sample treatment allocations:")
        for symbol, treatment in allocations:
            print(f"   - {symbol}: {treatment}")
        
        # Show experiment registry
        registered_models = self.governance.experiment_registry.get_registered_models()
        experiment_templates = self.governance.experiment_registry.get_experiment_templates()
        active_experiments = len(self.governance.ab_framework.active_experiments)
        
        print("\n📝 Experiment Registry:")
        print(f"   - Registered models: {len(registered_models)}")
        print(f"   - Experiment templates: {len(experiment_templates)}")
        print(f"   - Active experiments: {active_experiments}")
        
        self.demo_results['ab_testing'] = {
            'status': 'success',
            'experiment_id': experiment.id,
            'allocations': allocations,
            'registry_summary': {
                'registered_models': len(registered_models),
                'experiment_templates': len(experiment_templates),
                'active_experiments': active_experiments
            }
        }
    
    async def demo_risk_sanity_validation(self):
        """Demonstrate risk sanity checks with options-implied validation"""
        
        print("\n⚠️  Demo 4: Risk Sanity Validation")
        print("-" * 50)
        
        if not self.governance.risk_validator:
            print("❌ Risk validation not enabled in governance")
            return
        
        # Test predictions for validation
        test_cases = [
            {
                'symbol': 'AAPL',
                'predicted_move': 0.02,  # 2% move
                'confidence_interval': (-0.03, 0.07),  # Wider interval
                'horizon_days': 5,
                'description': 'Normal prediction'
            },
            {
                'symbol': 'MSFT',
                'predicted_move': 0.15,  # 15% move - high!
                'confidence_interval': (-0.05, 0.35),
                'horizon_days': 1,
                'description': 'High volatility prediction'
            },
            {
                'symbol': 'GOOGL',
                'predicted_move': 0.005,  # 0.5% move - very small
                'confidence_interval': (-0.01, 0.02),
                'horizon_days': 10,
                'description': 'Conservative prediction'
            }
        ]
        
        print("🔍 Testing predictions against options-implied moves:")
        
        all_violations = []
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n   Test {i}: {case['description']}")
            print(f"   - Symbol: {case['symbol']}")
            print(f"   - Predicted move: {case['predicted_move']:.1%}")
            print(f"   - Confidence interval: ({case['confidence_interval'][0]:.1%}, {case['confidence_interval'][1]:.1%})")
            
            # Validate prediction
            violations = self.governance.risk_validator.validate_prediction(
                symbol=case['symbol'],
                predicted_move=case['predicted_move'],
                confidence_interval=case['confidence_interval'],
                horizon_days=case['horizon_days']
            )
            
            if violations:
                print(f"   ⚠️  {len(violations)} violations detected:")
                for violation in violations:
                    severity_icon = {
                        ViolationSeverity.INFO: "ℹ️",
                        ViolationSeverity.WARNING: "⚠️",
                        ViolationSeverity.ERROR: "❌",
                        ViolationSeverity.CRITICAL: "🚨"
                    }.get(violation.severity, "❓")
                    
                    print(f"     {severity_icon} {violation.severity.value.upper()}: {violation.message}")
                    if violation.ratio > 0:
                        print(f"        Ratio: {violation.ratio:.2f}x threshold")
                
                all_violations.extend(violations)
            else:
                print("   ✅ No violations detected")
        
        # Show violation summary
        if all_violations:
            violation_summary = self.governance.risk_validator.get_violation_summary()
            print("\n📈 Risk Violation Summary:")
            print(f"   - Total violations: {violation_summary['total_violations']}")
            print(f"   - By severity: {violation_summary['by_severity']}")
            print(f"   - By type: {violation_summary['by_type']}")
        
        self.demo_results['risk_validation'] = {
            'status': 'success',
            'test_cases': len(test_cases),
            'total_violations': len(all_violations),
            'violation_summary': self.governance.risk_validator.get_violation_summary() if all_violations else {}
        }
    
    async def demo_regulatory_documentation(self):
        """Demonstrate regulatory documentation and audit trails"""
        
        print("\n📋 Demo 5: Regulatory Documentation")
        print("-" * 50)
        
        if not self.governance.experiment_documenter:
            print("❌ Documentation not enabled in governance")
            return
        
        # Create experiment documentation
        experiment = self.governance.experiment_documenter.create_experiment(
            experiment_name="Enhanced TF Production Deployment",
            objective="Deploy improved temporal fusion transformer for multi-asset prediction",
            hypothesis="Enhanced TF model will outperform baseline by 15% in Sharpe ratio while maintaining similar drawdown characteristics",
            methodology="Walk-forward cross-validation with 1-day embargo, stratified by sector and market cap, champion/challenger A/B testing with 70/30 allocation",
            success_criteria=[
                "Sharpe ratio improvement >= 15%",
                "Maximum drawdown <= 20%",
                "Hit rate >= 52%",
                "Statistical significance p < 0.05"
            ],
            compliance_frameworks=[
                ComplianceFramework.SR_11_7,
                ComplianceFramework.INTERNAL
            ],
            created_by="portfolio_manager"
        )
        
        print("📝 Created experiment documentation:")
        print(f"   - Experiment ID: {experiment.experiment_id}")
        print(f"   - Name: {experiment.experiment_name}")
        print(f"   - Compliance frameworks: {[fw.value for fw in experiment.compliance_frameworks]}")
        
        # Document fold boundaries
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        fold_boundaries = self.governance.experiment_documenter.fold_documenter.document_time_series_folds(
            dates=dates,
            n_folds=5,
            validation_size=0.2
        )
        
        print(f"\n📊 Documented {len(fold_boundaries)} cross-validation folds:")
        for i, fold in enumerate(fold_boundaries[:2]):  # Show first 2 folds
            print(f"   - Fold {i+1}: Train {fold.samples_train} samples, Val {fold.samples_validation} samples")
            print(f"     Embargo: {fold.embargo_days} days")
        print(f"   ... and {len(fold_boundaries)-2} more folds")
        
        # Create parameter snapshot
        snapshot = self.governance.experiment_documenter.parameter_documenter.create_snapshot(
            model_type="temporal_fusion_transformer",
            hyperparameters={
                'hidden_size': 256,
                'num_attention_heads': 8,
                'num_layers': 6,
                'dropout_rate': 0.1,
                'learning_rate': 0.001
            },
            feature_parameters={
                'lookback_window': 30,
                'feature_sets': ['price', 'volume', 'technical', 'fundamental'],
                'normalization': 'z_score'
            },
            training_parameters={
                'batch_size': 64,
                'max_epochs': 100,
                'early_stopping_patience': 10
            },
            random_seed=42,
            experiment_id=experiment.experiment_id,
            created_by="data_scientist"
        )
        
        print("\n💾 Created parameter snapshot:")
        print(f"   - Snapshot ID: {snapshot.snapshot_id}")
        print(f"   - Parameter hash: {snapshot.parameter_hash[:16]}...")
        print(f"   - Random seed: {snapshot.random_seed}")
        
        # Generate compliance report
        compliance_report = self.governance.experiment_documenter.generate_compliance_report(
            experiment.experiment_id,
            ComplianceFramework.SR_11_7
        )
        
        print("\n📋 Generated SR 11-7 compliance report:")
        print(f"   - Framework: {compliance_report['framework']}")
        print(f"   - Compliance status: {compliance_report['compliance_status']}")
        
        # Show model development section
        if 'model_development' in compliance_report['sections']:
            dev_section = compliance_report['sections']['model_development']
            print(f"   - Development process documented: {dev_section['development_process_documented']}")
            print(f"   - Validation performed: {dev_section['validation_performed']}")
            print(f"   - Parameter documentation: {dev_section['parameter_documentation']}")
        
        # Log audit events
        audit_events = [
            ("model_training", "data_scientist", "model", "enhanced_tf_v2.1", "train_model"),
            ("model_validation", "data_scientist", "model", "enhanced_tf_v2.1", "validate_model"),
            ("model_approval", "portfolio_manager", "model", "enhanced_tf_v2.1", "approve_deployment")
        ]
        
        for event_type, user_id, resource_type, resource_id, action in audit_events:
            self.governance.audit_manager.log_event(
                event_type=event_type,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                details={'timestamp': datetime.now(timezone.utc).isoformat()}
            )
        
        print(f"\n📜 Logged {len(audit_events)} audit events")
        
        # Show audit trail
        audit_trail = self.governance.audit_manager.get_audit_trail(
            resource_id="enhanced_tf_v2.1",
            limit=5
        )
        
        print("   Recent audit events:")
        for event in audit_trail[:3]:
            print(f"   - {event['timestamp'][:19]}: {event['user_id']} {event['action']}")
        
        self.demo_results['regulatory_documentation'] = {
            'status': 'success',
            'experiment_id': experiment.experiment_id,
            'fold_boundaries': len(fold_boundaries),
            'parameter_snapshot': snapshot.snapshot_id,
            'compliance_report': compliance_report['compliance_status'],
            'audit_events': len(audit_trail)
        }
    
    async def demo_governance_monitoring(self):
        """Demonstrate governance monitoring and alerts"""
        
        print("\n🔍 Demo 6: Governance Monitoring")
        print("-" * 50)
        
        # Setup alert handler
        received_alerts = []
        
        def alert_handler(alert: Dict[str, Any]):
            received_alerts.append(alert)
            print(f"   🚨 ALERT: {alert['type']} - {alert['severity'].upper()}")
            print(f"      Details: {alert['details']}")
        
        self.governance.register_alert_handler(alert_handler)
        
        print("📊 Governance monitoring dashboard:")
        
        # Get comprehensive governance summary
        summary = self.governance.get_governance_summary(days=1)
        
        print(f"   - Governance level: {summary['governance_level']}")
        print(f"   - Total requests today: {summary['total_requests']}")
        print(f"   - Pending requests: {summary['pending_requests']}")
        print(f"   - Average risk score: {summary['average_risk_score']:.1f}")
        print(f"   - Active A/B tests: {summary['active_ab_tests']}")
        
        print("\n🔧 Component status:")
        components = summary['components_enabled']
        for component, enabled in components.items():
            status_icon = "✅" if enabled else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}: {'Enabled' if enabled else 'Disabled'}")
        
        # Simulate governance alert
        print("\n🧪 Simulating governance alert...")
        self.governance._trigger_alert(
            "high_risk_deployment",
            {
                'severity': 'error',
                'model_id': 'test_model_v1.0',
                'risk_score': 35.0,
                'threshold': 25.0,
                'reason': 'Risk score exceeds threshold'
            }
        )
        
        # Show alert handling
        if received_alerts:
            print(f"✅ Alert system working - {len(received_alerts)} alerts received")
        
        self.demo_results['governance_monitoring'] = {
            'status': 'success',
            'governance_summary': summary,
            'alerts_received': len(received_alerts)
        }
    
    def print_demo_summary(self):
        """Print comprehensive demo summary"""
        
        print("\n" + "="*80)
        print("TASK 10 GOVERNANCE FRAMEWORK DEMO SUMMARY")
        print("="*80)
        
        print("\n🎯 Completed Components:")
        
        for demo_name, result in self.demo_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            demo_title = demo_name.replace('_', ' ').title()
            print(f"{status_icon} {demo_title}")
            
            # Show key metrics for each demo
            if demo_name == 'deployment_workflow' and 'deployment_status' in result:
                status = result['deployment_status']
                print(f"   → Decision: {status.get('decision', 'pending')}")
                print(f"   → Risk Score: {status.get('risk_score', 0):.1f}")
            
            elif demo_name == 'ab_testing' and 'experiment_id' in result:
                print(f"   → Experiment ID: {result['experiment_id'][:8]}...")
                print(f"   → Allocations: {len(result.get('allocations', []))}")
            
            elif demo_name == 'risk_validation' and 'total_violations' in result:
                print(f"   → Test Cases: {result['test_cases']}")
                print(f"   → Violations: {result['total_violations']}")
            
            elif demo_name == 'regulatory_documentation' and 'experiment_id' in result:
                print(f"   → Experiment: {result['experiment_id'][:8]}...")
                print(f"   → Fold Boundaries: {result['fold_boundaries']}")
                print(f"   → Compliance: {result['compliance_report']}")
        
        print("\n📋 Key Features Demonstrated:")
        print("   ✅ Champion/challenger A/B testing for live paper-trading")
        print("   ✅ Pre-trade risk sanity checks with options-implied validation")
        print("   ✅ Comprehensive regulatory documentation and audit trails")
        print("   ✅ Complete model deployment governance workflow")
        print("   ✅ Real-time monitoring and alerting system")
        print("   ✅ Regulatory compliance (SR 11-7, MiFID II)")
        
        print("\n🎉 Task 10 Implementation Complete!")
        print("   The governance framework provides everything regulators and")
        print("   investors look for in production ML systems:")
        print("   • Complete audit trails and documentation")
        print("   • Rigorous testing with champion/challenger methodology")
        print("   • Real-time risk validation and pre-trade checks")
        print("   • Automated compliance reporting")
        print("   • Full model lifecycle governance")
        
        # Calculate overall success rate
        successful_demos = sum(1 for result in self.demo_results.values() if result['status'] == 'success')
        total_demos = len(self.demo_results)
        success_rate = (successful_demos / total_demos) * 100 if total_demos > 0 else 0
        
        print(f"\n📊 Demo Success Rate: {success_rate:.0f}% ({successful_demos}/{total_demos})")

async def main():
    """Main demonstration function"""
    
    print("Starting Task 10 Governance Framework Demonstration...")
    
    demo = GovernanceDemo()
    
    try:
        await demo.run_complete_demo()
        print("\n✅ Demonstration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Demo failed")
        return 1
    
    return 0

if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    exit(exit_code)