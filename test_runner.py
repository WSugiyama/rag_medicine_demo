"""
Simple test runner for RAG system evaluation
"""

import re
import sys
import os
import argparse
from typing import List, Dict
import json
from datetime import datetime
import unittest

# Import test modules
from test_config import *


class RAGSystemTest(unittest.TestCase):
    """Test suite for RAG system functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.results = []
    
    def print_result(self, test_name: str, status: str, details: str = "", error: str = ""):
        """Print formatted test result"""
        status_symbol = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_symbol} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        if error:
            print(f"   Error: {error}")
        print()
    
    def test_covid_term_extraction(self):
        """Test COVID-specific query term extraction"""
        print("Running COVID-19 Query Tests...")
        print("-" * 40)
        
        try:
            from rag_pubmed_assistant import extract_medical_terms
            
            for i, query in enumerate(COVID_QUERIES, 1):
                print(f"Test {i}/{len(COVID_QUERIES)}: {query[:50]}...")
                
                # Extract medical terms
                terms = extract_medical_terms(query)
                
                # Assert that terms were extracted
                self.assertIsNotNone(terms, f"No terms extracted for query: {query}")
                self.assertIsInstance(terms, list, f"Terms should be a list, got {type(terms)}")
                self.assertGreater(len(terms), 0, f"No terms found for query: {query}")
                
                # Check if COVID terms are extracted
                covid_found = any(term in ['covid', 'coronavirus', 'sars-cov'] 
                                for term in ' '.join(terms).lower().split())
                
                if covid_found:
                    self.print_result(f"COVID Test {i}", "PASS", f"Terms: {terms}")
                    self.results.append({
                        "query": query,
                        "extracted_terms": terms,
                        "covid_terms_found": covid_found,
                        "status": "PASS"
                    })
                else:
                    self.print_result(f"COVID Test {i}", "FAIL", f"Terms: {terms}", "No COVID terms found")
                    self.results.append({
                        "query": query,
                        "extracted_terms": terms,
                        "covid_terms_found": covid_found,
                        "status": "FAIL"
                    })
                
        except Exception as e:
            self.fail(f"COVID term extraction test failed with error: {str(e)}")
    
    def test_citation_handling(self):
        """Test citation handling functionality"""
        print("Running Citation Tests...")
        print("-" * 40)
        
        try:
            from rag_pubmed_assistant import add_citations_to_answer, remap_citations_in_answer
            
            # Test citation addition
            test_answer = "COVID-19 affects immunity. Probiotics may help."
            test_sources = ["Source 1", "Source 2"]
            
            result = add_citations_to_answer(test_answer, test_sources)
            
            # Assert citations were added
            self.assertIsNotNone(result, "Citation addition should return a result")
            self.assertIsInstance(result, str, "Result should be a string")
            self.assertGreater(len(result), len(test_answer), "Result should be longer than input")
            
            # Check for citation pattern
            has_citations = bool(re.search(r'\[\d+\]', result))
            self.assertTrue(has_citations, "Result should contain citations")
            
            if has_citations:
                self.print_result("Citation Addition", "PASS", f"Result: {result}")
            else:
                self.print_result("Citation Addition", "FAIL", f"Result: {result}", "No citations found")
            
            # Test citation remapping
            test_answer_with_citations = "Statement [2]. Another [1]. Final [3]."
            mapping = {1: 3, 2: 1, 3: 2}
            expected_result = "Statement [1]. Another [3]. Final [2]."
            
            remapped_result = remap_citations_in_answer(test_answer_with_citations, mapping)
            
            # Assert remapping worked correctly
            self.assertEqual(remapped_result, expected_result, 
                           f"Citation remapping failed. Expected: {expected_result}, Got: {remapped_result}")
            
            self.print_result("Citation Remapping", "PASS", f"Result: {remapped_result}")
            
        except ImportError as e:
            self.skipTest(f"Could not import required functions: {e}")
        except Exception as e:
            self.fail(f"Citation handling test failed with error: {str(e)}")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        print("Running Performance Benchmarks...")
        print("-" * 40)
        
        try:
            from rag_pubmed_assistant import extract_medical_terms
            
            all_queries = COVID_QUERIES + GENERAL_MEDICAL_QUERIES
            term_success = 0
            
            for query in all_queries:
                terms = extract_medical_terms(query)
                # Basic check - should extract at least one medical-related term
                if terms and len(terms) > 0:
                    term_success += 1
            
            accuracy = term_success / max(1, len(all_queries))
            
            # Assert minimum performance threshold
            self.assertGreaterEqual(accuracy, PERFORMANCE_BENCHMARKS["term_extraction_accuracy"], 
                                  f"Term extraction accuracy {accuracy:.1%} below threshold {PERFORMANCE_BENCHMARKS['term_extraction_accuracy']:.1%}")
            
            self.print_result("Performance Benchmark", "PASS", 
                            f"Term extraction success: {term_success}/{len(all_queries)} ({accuracy:.1%})")
            
        except Exception as e:
            self.fail(f"Performance benchmark test failed with error: {str(e)}")
    
    def test_medical_term_extraction_edge_cases(self):
        """Test edge cases and out-of-domain filtering for medical term extraction"""
        print("Running Edge Case Tests...")
        print("-" * 40)
        
        try:
            from rag_pubmed_assistant import extract_medical_terms
            
            # Test empty query
            empty_terms = extract_medical_terms("")
            self.assertEqual(empty_terms, [], "Empty query should return an empty list")
            
            # Test query with no medical terms (English & Indonesian)
            non_medical_en = extract_medical_terms("What is the weather like today?")
            non_medical_id = extract_medical_terms("Bagaimana cara membuat kue bolu?")
            
            self.assertEqual(non_medical_en, [], "Non-medical English query should return an empty list")
            self.assertEqual(non_medical_id, [], "Non-medical Indonesian query should return an empty list")
            
            # Test very long query containing medical terms
            long_query = "What are the effects of " + "very " * 100 + "long COVID-19 symptoms?"
            long_terms = extract_medical_terms(long_query)
            self.assertIn("covid", [term.lower() for term in long_terms], "Long query should extract COVID terms")
            
            self.print_result("Edge Cases", "PASS", "All edge cases and out-of-domain gates handled correctly")
            
        except Exception as e:
            self.fail(f"Edge case test failed with error: {str(e)}")

    
    def test_citation_patterns(self):
        """Test citation pattern validation"""
        print("Running Citation Pattern Tests...")
        print("-" * 40)
        
        try:
            from rag_pubmed_assistant import add_citations_to_answer
            
            # Test various citation patterns
            test_cases = [
                ("Single sentence.", ["Source 1"]),
                ("First sentence. Second sentence.", ["Source 1", "Source 2"]),
                ("Multiple. Sentences. Here.", ["Source 1", "Source 2", "Source 3"])
            ]
            
            for i, (text, sources) in enumerate(test_cases):
                result = add_citations_to_answer(text, sources)
                
                # Assert result contains citations
                citation_count = len(re.findall(r'\[\d+\]', result))
                self.assertGreater(citation_count, 0, f"No citations found in test case {i+1}")
                
                # Assert citation numbers are sequential
                citations = re.findall(r'\[(\d+)\]', result)
                expected_numbers = list(range(1, len(citations) + 1))
                actual_numbers = [int(c) for c in citations]
                self.assertEqual(actual_numbers, expected_numbers, 
                               f"Citation numbers should be sequential: expected {expected_numbers}, got {actual_numbers}")
            
            self.print_result("Citation Patterns", "PASS", "All citation patterns validated")
            
        except Exception as e:
            self.fail(f"Citation pattern test failed with error: {str(e)}")

    def test_indonesian_term_extraction(self):
        """Test multilingual (Indonesian) medical term extraction"""
        print("Running Indonesian Query Tests...")
        print("-" * 40)
        
        try:
            from rag_pubmed_assistant import extract_medical_terms
            
            for i, query in enumerate(INDONESIAN_MEDICAL_QUERIES, 1):
                terms = extract_medical_terms(query)
                
                # Basic assertions for Indonesian support
                self.assertIsNotNone(terms, f"No terms extracted for query: {query}")
                self.assertIsInstance(terms, list, "Extracted terms must be a list")
                self.assertGreater(len(terms), 0, f"Failed to extract keywords for Indonesian query: {query}")
                
                self.print_result(f"Indonesian Test {i}", "PASS", f"Query: '{query}' -> Terms: {terms}")
                
        except Exception as e:
            self.fail(f"Indonesian term extraction test failed with error: {str(e)}")


def run_covid_tests():
    """Run COVID-specific query tests (legacy function)"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(RAGSystemTest)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return test_result.wasSuccessful()


def run_citation_tests():
    """Run citation handling tests (legacy function)"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(RAGSystemTest)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return test_result.wasSuccessful()


def run_performance_benchmark():
    """Run performance benchmarks (legacy function)"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(RAGSystemTest)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    return test_result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument("--suite", choices=["covid", "citations", "perf", "indonesian", "all"], default="all")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    if args.suite in ("covid", "all"):
        test_suite.addTest(RAGSystemTest("test_covid_term_extraction"))
    if args.suite in ("citations", "all"):
        test_suite.addTest(RAGSystemTest("test_citation_handling"))
    if args.suite in ("perf", "all"):
        test_suite.addTest(RAGSystemTest("test_performance_benchmarks"))

    if args.suite in ("indonesian", "all"):
        test_suite.addTest(RAGSystemTest("test_indonesian_term_extraction"))
    if args.suite == "all":
        test_suite.addTest(RAGSystemTest("test_medical_term_extraction_edge_cases"))
        test_suite.addTest(RAGSystemTest("test_citation_patterns"))

    
    # Add edge case tests for comprehensive coverage
    if args.suite == "all":
        test_suite.addTest(RAGSystemTest("test_medical_term_extraction_edge_cases"))
        test_suite.addTest(RAGSystemTest("test_citation_patterns"))
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    test_result = unittest.TextTestRunner(verbosity=verbosity).run(test_suite)
    
    # Generate report
    total_tests = test_result.testsRun
    failed_tests = len(test_result.failures)
    error_tests = len(test_result.errors)
    successful_tests = total_tests - failed_tests - error_tests
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Errors: {error_tests}")
    print(f"Success Rate: {successful_tests/total_tests:.1%}" if total_tests > 0 else "Success Rate: N/A")
    
    # Save results to JSON
    try:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        results_data = {
            "timestamp": timestamp,
            "total_tests": total_tests,
            "successful": successful_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": successful_tests/total_tests if total_tests > 0 else 0,
            "failures": [{"test": str(f[0]), "error": str(f[1])} for f in test_result.failures],
            "errors": [{"test": str(e[0]), "error": str(e[1])} for e in test_result.errors]
        }
        
        with open(f"test_results_{timestamp}.json", "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: test_results_{timestamp}.json")
        
    except Exception as e:
        print(f"\nCould not save results: {e}")
    
    return test_result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


