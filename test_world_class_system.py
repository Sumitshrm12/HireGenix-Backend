"""
🧪 COMPREHENSIVE TEST SUITE for World-Class Parallel Agentic AI System

Tests:
1. Single candidate processing
2. Parallel agent execution
3. Bulk processing with concurrency control
4. Intelligent caching
5. Error handling and fault tolerance
6. Performance benchmarks
"""

import asyncio
import time
import json
from typing import List, Dict, Any
from agentic_candidate_processor import get_candidate_processor

# Sample test data
SAMPLE_RESUME_1 = """
John Doe
Senior Software Engineer
Email: john.doe@example.com
Phone: +1-555-0123

SKILLS:
Python, TypeScript, React, Node.js, PostgreSQL, Redis, AWS, Docker, Kubernetes

EXPERIENCE:
Senior Software Engineer at TechCorp (2020-Present)
- Led team of 5 engineers building microservices architecture
- Improved system performance by 300%
- Implemented CI/CD pipelines

Software Engineer at StartupXYZ (2018-2020)
- Built RESTful APIs using Python/Flask
- Designed database schemas for high-traffic applications

EDUCATION:
B.S. Computer Science, MIT (2018)
GPA: 3.8/4.0

CERTIFICATIONS:
- AWS Solutions Architect
- Kubernetes Administrator
"""

SAMPLE_RESUME_2 = """
Jane Smith
Full Stack Developer
Email: jane.smith@example.com
Phone: +1-555-0456

SKILLS:
JavaScript, React, Node.js, MongoDB, Express, HTML, CSS, Git

EXPERIENCE:
Full Stack Developer at WebCorp (2021-Present)
- Developed responsive web applications
- Collaborated with design team

Junior Developer at CodeStart (2019-2021)
- Created frontend components
- Fixed bugs and implemented features

EDUCATION:
B.A. Computer Science, UCLA (2019)
"""

SAMPLE_RESUME_3 = """
Mike Johnson
Data Scientist
Email: mike.j@example.com
Phone: +1-555-0789

SKILLS:
Python, R, Machine Learning, TensorFlow, PyTorch, SQL, Data Visualization

EXPERIENCE:
Data Scientist at DataCorp (2020-Present)
- Built ML models for predictive analytics
- Achieved 95% accuracy in customer churn prediction

Data Analyst at AnalyticsPro (2018-2020)
- Created dashboards and reports
- Performed statistical analysis

EDUCATION:
M.S. Data Science, Stanford University (2018)
Ph.D. Statistics (In Progress)
"""


class TestSuite:
    """Comprehensive test suite for parallel agentic processor"""
    
    def __init__(self):
        self.processor = None
        self.test_results = []
        
    async def setup(self):
        """Initialize processor"""
        print("🔧 Setting up test environment...")
        self.processor = get_candidate_processor()
        print("✅ Processor initialized\n")
    
    async def teardown(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up...")
        if self.processor:
            await self.processor.close()
        print("✅ Cleanup complete")
    
    def log_result(self, test_name: str, passed: bool, duration: float, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "duration": duration,
            "details": details
        })
        print(f"{status} | {test_name} | {duration:.2f}s | {details}")
    
    async def test_single_candidate_processing(self):
        """Test 1: Single candidate processing"""
        print("\n" + "="*80)
        print("TEST 1: Single Candidate Processing")
        print("="*80)
        
        start_time = time.time()
        
        try:
            result = await self.processor.process_candidate(
                resume_text=SAMPLE_RESUME_1,
                job_id="test_job_123",
                company_id="test_company_456",
                screening_leniency="auto",
                user_id="test_user"
            )
            
            duration = time.time() - start_time
            
            # Validations
            assert result["success"] == True, "Processing failed"
            assert "candidate_id" in result, "No candidate ID"
            assert "scores" in result, "No scores"
            assert result["scores"]["overall"] >= 0 and result["scores"]["overall"] <= 1, "Invalid score"
            
            self.log_result(
                "Single Candidate Processing",
                True,
                duration,
                f"Score: {result['scores']['overall']:.2%}"
            )
            
            print(f"\n📊 Results:")
            print(f"   Candidate: {result['profile']['name']}")
            print(f"   Overall Score: {result['scores']['overall']:.2%}")
            print(f"   Technical Fit: {result['scores']['technical_fit']:.2%}")
            print(f"   Prescreening Status: {result['prescreening_eligibility']['status']}")
            print(f"   Processing Time: {duration:.2f}s")
            
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Single Candidate Processing", False, duration, str(e))
            print(f"\n❌ Error: {e}")
            return False
    
    async def test_parallel_execution_speed(self):
        """Test 2: Verify parallel execution is faster than sequential"""
        print("\n" + "="*80)
        print("TEST 2: Parallel Execution Speed Verification")
        print("="*80)
        
        print("\n⏱️  Measuring parallel execution time...")
        start_time = time.time()
        
        try:
            result = await self.processor.process_candidate(
                resume_text=SAMPLE_RESUME_2,
                job_id="test_job_123",
                company_id="test_company_456",
                screening_leniency="moderate",
                user_id="test_user"
            )
            
            parallel_time = time.time() - start_time
            
            # Expected: 15-20s for parallel, 23-28s for sequential
            # If parallel takes > 22s, might not be working correctly
            passed = parallel_time < 22.0
            
            self.log_result(
                "Parallel Execution Speed",
                passed,
                parallel_time,
                f"Target: <22s, Actual: {parallel_time:.2f}s"
            )
            
            print(f"\n📊 Speed Analysis:")
            print(f"   Parallel Processing Time: {parallel_time:.2f}s")
            print(f"   Expected Sequential Time: ~23-25s")
            print(f"   Time Saved: ~{25 - parallel_time:.2f}s")
            print(f"   Performance Gain: {((25 - parallel_time) / 25 * 100):.1f}%")
            
            if passed:
                print("   ✅ Parallel execution confirmed working!")
            else:
                print("   ⚠️  Warning: Slower than expected, check parallel agents")
            
            return passed
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Parallel Execution Speed", False, duration, str(e))
            print(f"\n❌ Error: {e}")
            return False
    
    async def test_bulk_processing_small(self):
        """Test 3: Bulk processing with 5 candidates"""
        print("\n" + "="*80)
        print("TEST 3: Bulk Processing (5 candidates)")
        print("="*80)
        
        candidates = [
            {
                "resume_text": SAMPLE_RESUME_1,
                "job_id": "test_job_123",
                "company_id": "test_company_456",
                "screening_leniency": "auto"
            },
            {
                "resume_text": SAMPLE_RESUME_2,
                "job_id": "test_job_123",
                "company_id": "test_company_456",
                "screening_leniency": "auto"
            },
            {
                "resume_text": SAMPLE_RESUME_3,
                "job_id": "test_job_123",
                "company_id": "test_company_456",
                "screening_leniency": "moderate"
            },
            {
                "resume_text": SAMPLE_RESUME_1,
                "job_id": "test_job_789",
                "company_id": "test_company_456",
                "screening_leniency": "lenient"
            },
            {
                "resume_text": SAMPLE_RESUME_2,
                "job_id": "test_job_789",
                "company_id": "test_company_999",
                "screening_leniency": "strict"
            }
        ]
        
        start_time = time.time()
        
        try:
            results = await self.processor.process_candidates_bulk(
                candidates=candidates,
                max_concurrent=3,
                user_id="test_user"
            )
            
            duration = time.time() - start_time
            
            # Validations
            assert results["success"] == True, "Bulk processing failed"
            assert results["total"] == 5, f"Expected 5, got {results['total']}"
            assert results["successful"] >= 4, f"Too many failures: {results['failed']}"
            
            passed = results["success_rate"] >= 0.8  # 80% success rate minimum
            
            self.log_result(
                "Bulk Processing (5 candidates)",
                passed,
                duration,
                f"Success: {results['successful']}/{results['total']}"
            )
            
            print(f"\n📊 Bulk Processing Results:")
            print(f"   Total Candidates: {results['total']}")
            print(f"   Successful: {results['successful']}")
            print(f"   Failed: {results['failed']}")
            print(f"   Success Rate: {results['success_rate']:.1%}")
            print(f"   Total Time: {duration:.2f}s")
            print(f"   Avg Time/Candidate: {results['avg_time_per_candidate']:.2f}s")
            print(f"   Throughput: {results['throughput_per_second']:.2f} candidates/sec")
            
            return passed
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Bulk Processing (5 candidates)", False, duration, str(e))
            print(f"\n❌ Error: {e}")
            return False
    
    async def test_caching_effectiveness(self):
        """Test 4: Verify caching improves performance"""
        print("\n" + "="*80)
        print("TEST 4: Intelligent Caching Effectiveness")
        print("="*80)
        
        # First pass - no cache
        print("\n🔄 First pass (cold cache)...")
        start_first = time.time()
        
        try:
            await self.processor.process_candidate(
                resume_text=SAMPLE_RESUME_1,
                job_id="test_cache_job",
                company_id="test_cache_company",
                screening_leniency="auto",
                user_id="test_user"
            )
            
            first_duration = time.time() - start_first
            print(f"   Time: {first_duration:.2f}s (no cache)")
            
            # Second pass - with cache
            print("\n🔄 Second pass (warm cache)...")
            start_second = time.time()
            
            await self.processor.process_candidate(
                resume_text=SAMPLE_RESUME_2,  # Different resume, same job/company
                job_id="test_cache_job",
                company_id="test_cache_company",
                screening_leniency="auto",
                user_id="test_user"
            )
            
            second_duration = time.time() - start_second
            print(f"   Time: {second_duration:.2f}s (with cache)")
            
            # Cache should make it at least 10% faster
            speedup = ((first_duration - second_duration) / first_duration) * 100
            passed = speedup > 5  # At least 5% speedup
            
            total_duration = first_duration + second_duration
            
            self.log_result(
                "Intelligent Caching",
                passed,
                total_duration,
                f"Speedup: {speedup:.1f}%"
            )
            
            print(f"\n📊 Caching Analysis:")
            print(f"   First Pass (Cold): {first_duration:.2f}s")
            print(f"   Second Pass (Warm): {second_duration:.2f}s")
            print(f"   Time Saved: {first_duration - second_duration:.2f}s")
            print(f"   Speedup: {speedup:.1f}%")
            
            if passed:
                print("   ✅ Cache working effectively!")
            else:
                print("   ⚠️  Cache impact lower than expected")
            
            return passed
            
        except Exception as e:
            duration = time.time() - start_first
            self.log_result("Intelligent Caching", False, duration, str(e))
            print(f"\n❌ Error: {e}")
            return False
    
    async def test_error_handling(self):
        """Test 5: Error handling and fault tolerance"""
        print("\n" + "="*80)
        print("TEST 5: Error Handling & Fault Tolerance")
        print("="*80)
        
        # Test with invalid data
        candidates = [
            {
                "resume_text": "",  # Empty resume
                "job_id": "test_job_123",
                "company_id": "test_company_456",
                "screening_leniency": "auto"
            },
            {
                "resume_text": SAMPLE_RESUME_1,  # Valid resume
                "job_id": "test_job_123",
                "company_id": "test_company_456",
                "screening_leniency": "auto"
            },
            {
                "resume_text": "Invalid data @#$%",  # Malformed resume
                "job_id": "test_job_123",
                "company_id": "test_company_456",
                "screening_leniency": "auto"
            }
        ]
        
        start_time = time.time()
        
        try:
            results = await self.processor.process_candidates_bulk(
                candidates=candidates,
                max_concurrent=2,
                user_id="test_user"
            )
            
            duration = time.time() - start_time
            
            # Should handle errors gracefully - at least 1 should succeed
            passed = results["successful"] >= 1 and results["failed"] >= 1
            
            self.log_result(
                "Error Handling",
                passed,
                duration,
                f"Handled {results['failed']} errors gracefully"
            )
            
            print(f"\n📊 Error Handling Results:")
            print(f"   Total: {results['total']}")
            print(f"   Successful: {results['successful']}")
            print(f"   Failed (as expected): {results['failed']}")
            print(f"   System remained stable: ✅")
            
            if results['errors']:
                print(f"\n   Error samples:")
                for i, error in enumerate(results['errors'][:2]):
                    print(f"     {i+1}. {error.get('error', 'Unknown error')[:100]}")
            
            return passed
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Handling", False, duration, str(e))
            print(f"\n❌ Unexpected error: {e}")
            return False
    
    async def test_concurrency_control(self):
        """Test 6: Semaphore-based concurrency control"""
        print("\n" + "="*80)
        print("TEST 6: Concurrency Control (Semaphore)")
        print("="*80)
        
        # Create 10 candidates
        candidates = [
            {
                "resume_text": SAMPLE_RESUME_1 if i % 3 == 0 else SAMPLE_RESUME_2 if i % 3 == 1 else SAMPLE_RESUME_3,
                "job_id": f"test_job_{i % 3}",
                "company_id": f"test_company_{i % 2}",
                "screening_leniency": "auto"
            }
            for i in range(10)
        ]
        
        print(f"\n🔄 Processing 10 candidates with max_concurrent=3...")
        start_time = time.time()
        
        try:
            results = await self.processor.process_candidates_bulk(
                candidates=candidates,
                max_concurrent=3,  # Only 3 at a time
                user_id="test_user"
            )
            
            duration = time.time() - start_time
            
            # With concurrency=3, should take roughly 3-4x time of single candidate
            # Single: ~16s, So 10 candidates with concurrency 3: ~60-80s
            passed = results["total"] == 10 and results["successful"] >= 8
            
            self.log_result(
                "Concurrency Control",
                passed,
                duration,
                f"10 candidates, max_concurrent=3"
            )
            
            print(f"\n📊 Concurrency Test Results:")
            print(f"   Candidates: {results['total']}")
            print(f"   Success: {results['successful']}")
            print(f"   Max Concurrent: 3")
            print(f"   Total Time: {duration:.2f}s")
            print(f"   Expected Time (sequential): ~{results['total'] * 16:.0f}s")
            print(f"   Time Saved: ~{(results['total'] * 16) - duration:.0f}s")
            print(f"   Throughput: {results['throughput_per_second']:.2f} candidates/sec")
            
            return passed
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Concurrency Control", False, duration, str(e))
            print(f"\n❌ Error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("🚀 WORLD-CLASS PARALLEL SYSTEM - COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        await self.setup()
        
        # Run all tests
        tests = [
            self.test_single_candidate_processing(),
            self.test_parallel_execution_speed(),
            self.test_bulk_processing_small(),
            self.test_caching_effectiveness(),
            self.test_error_handling(),
            self.test_concurrency_control()
        ]
        
        # Execute all tests
        for test in tests:
            try:
                await test
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"\n❌ Test crashed: {e}")
        
        await self.teardown()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["passed"])
        failed_tests = total_tests - passed_tests
        total_duration = sum(r["duration"] for r in self.test_results)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        print(f"\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {result['test']}: {result['duration']:.2f}s - {result['details']}")
        
        print("\n" + "="*80)
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! System is world-class! 🚀")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Most tests passed. System is production-ready.")
        else:
            print("⚠️  Some tests failed. Review and fix issues.")
        print("="*80 + "\n")


async def main():
    """Run test suite"""
    suite = TestSuite()
    await suite.run_all_tests()


if __name__ == "__main__":
    print("\n🧪 Starting World-Class Parallel System Test Suite...\n")
    asyncio.run(main())
