#!/usr/bin/env python3
"""
Smart Contract SAST Tool
A static analysis security testing tool for Solidity smart contracts
targeting OWASP Smart Contract Top 10 vulnerabilities.
"""

import re
import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class Finding:
    rule_id: str
    title: str
    description: str
    severity: Severity
    line_number: int
    code_snippet: str
    recommendation: str
    owasp_category: str

class SolidityParser:
    """Basic Solidity code parser for extracting relevant information"""
    
    def _init_(self, content: str):
        self.content = content
        self.lines = content.split('\n')
        self.functions = self._extract_functions()
        self.modifiers = self._extract_modifiers()
        self.state_variables = self._extract_state_variables()
        self.imports = self._extract_imports()
    
    def _extract_functions(self) -> List[Dict[str, Any]]:
        """Extract function definitions with their properties"""
        functions = []
        function_pattern = r'function\s+(\w+)\s*\([^)]\)\s(public|private|internal|external)?\s*(view|pure|payable)?\s*(returns\s*\([^)]\))?\s\{'
        
        for i, line in enumerate(self.lines):
            matches = re.finditer(function_pattern, line)
            for match in matches:
                func_info = {
                    'name': match.group(1),
                    'visibility': match.group(2) or 'internal',
                    'state_mutability': match.group(3) or '',
                    'returns': match.group(4) or '',
                    'line': i + 1,
                    'full_line': line.strip()
                }
                functions.append(func_info)
        
        return functions
    
    def _extract_modifiers(self) -> List[Dict[str, Any]]:
        """Extract modifier definitions"""
        modifiers = []
        modifier_pattern = r'modifier\s+(\w+)\s*\([^)]\)\s\{'
        
        for i, line in enumerate(self.lines):
            matches = re.finditer(modifier_pattern, line)
            for match in matches:
                mod_info = {
                    'name': match.group(1),
                    'line': i + 1,
                    'full_line': line.strip()
                }
                modifiers.append(mod_info)
        
        return modifiers
    
    def _extract_state_variables(self) -> List[Dict[str, Any]]:
        """Extract state variable declarations"""
        variables = []
        # Simple pattern for state variables (can be enhanced)
        var_pattern = r'(uint256|uint|int|bool|address|string|bytes\d*|mapping\([^)]+\))\s+(public|private|internal)?\s+(\w+)'
        
        for i, line in enumerate(self.lines):
            # Skip function bodies and comments
            if 'function' in line or line.strip().startswith('//'):
                continue
                
            matches = re.finditer(var_pattern, line)
            for match in matches:
                var_info = {
                    'type': match.group(1),
                    'visibility': match.group(2) or 'internal',
                    'name': match.group(3),
                    'line': i + 1,
                    'full_line': line.strip()
                }
                variables.append(var_info)
        
        return variables
    
    def _extract_imports(self) -> List[str]:
        """Extract import statements"""
        imports = []
        import_pattern = r'import\s+["\']([^"\']+)["\']'
        
        for line in self.lines:
            matches = re.finditer(import_pattern, line)
            for match in matches:
                imports.append(match.group(1))
        
        return imports

class VulnerabilityDetector:
    """Main vulnerability detection engine"""
    
    def _init_(self):
        self.rules = self._load_detection_rules()
    
    def _load_detection_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load vulnerability detection rules based on OWASP Smart Contract Top 10"""
        return {
            'SC01': {
                'name': 'Reentrancy',
                'owasp_category': 'SC01 - Reentrancy',
                'severity': Severity.CRITICAL,
                'patterns': [
                    r'\.call\s*\(',
                    r'\.send\s*\(',
                    r'\.transfer\s*\(',
                    r'address\([^)]+\)\.call',
                ],
                'description': 'Potential reentrancy vulnerability detected',
                'recommendation': 'Use the checks-effects-interactions pattern or reentrancy guards'
            },
            'SC02': {
                'name': 'Integer Overflow/Underflow',
                'owasp_category': 'SC02 - Integer Overflow and Underflow',
                'severity': Severity.HIGH,
                'patterns': [
                    r'\+\+|\-\-',
                    r'[a-zA-Z_]\w*\s*[\+\-\\/]\s=',
                    r'[a-zA-Z_]\w*\s*=\s*[a-zA-Z_]\w*\s*[\+\-\*\/]',
                ],
                'description': 'Potential integer overflow/underflow vulnerability',
                'recommendation': 'Use SafeMath library or Solidity ^0.8.0 built-in overflow protection'
            },
            'SC03': {
                'name': 'Unsafe External Calls',
                'owasp_category': 'SC03 - Unsafe External Calls',
                'severity': Severity.HIGH,
                'patterns': [
                    r'\.call\s*\(',
                    r'\.delegatecall\s*\(',
                    r'\.staticcall\s*\(',
                ],
                'description': 'Unsafe external call detected',
                'recommendation': 'Validate return values and use secure calling patterns'
            },
            'SC04': {
                'name': 'Access Control Issues',
                'owasp_category': 'SC04 - Access Control Issues',
                'severity': Severity.HIGH,
                'patterns': [
                    r'function\s+\w+\s*\([^)]\)\s*public\s(?!.*onlyOwner)(?!.*require)',
                    r'function\s+\w+\s*\([^)]\)\s*external\s(?!.*onlyOwner)(?!.*require)',
                ],
                'description': 'Public/external function without access control',
                'recommendation': 'Implement proper access control using modifiers or require statements'
            },
            'SC05': {
                'name': 'Denial of Service',
                'owasp_category': 'SC05 - Denial of Service',
                'severity': Severity.MEDIUM,
                'patterns': [
                    r'for\s*\([^)];\s[^;]\.length\s;',
                    r'while\s*\([^)]*\.length',
                    r'gas\(\)',
                ],
                'description': 'Potential denial of service through gas limit issues',
                'recommendation': 'Implement gas-efficient loops and avoid unbounded operations'
            },
            'SC06': {
                'name': 'Bad Randomness',
                'owasp_category': 'SC06 - Bad Randomness',
                'severity': Severity.MEDIUM,
                'patterns': [
                    r'block\.timestamp',
                    r'block\.difficulty',
                    r'block\.number',
                    r'blockhash\(',
                    r'keccak256\([^)]*block\.',
                ],
                'description': 'Insecure randomness source detected',
                'recommendation': 'Use secure randomness sources like Chainlink VRF'
            },
            'SC07': {
                'name': 'Front-Running',
                'owasp_category': 'SC07 - Front-Running',
                'severity': Severity.MEDIUM,
                'patterns': [
                    r'tx\.origin',
                    r'block\.timestamp.*==',
                ],
                'description': 'Potential front-running vulnerability',
                'recommendation': 'Use commit-reveal schemes or other anti-front-running mechanisms'
            },
            'SC08': {
                'name': 'Time Manipulation',
                'owasp_category': 'SC08 - Time Manipulation',
                'severity': Severity.MEDIUM,
                'patterns': [
                    r'now\s*[<>=]',
                    r'block\.timestamp\s*[<>=]',
                ],
                'description': 'Dangerous reliance on block timestamp',
                'recommendation': 'Avoid using block.timestamp for critical logic'
            },
            'SC09': {
                'name': 'Unchecked Return Values',
                'owasp_category': 'SC09 - Unchecked Low Level Calls',
                'severity': Severity.MEDIUM,
                'patterns': [
                    r'\.call\([^)]\)\s;',
                    r'\.send\([^)]\)\s;',
                    r'\.delegatecall\([^)]\)\s;',
                ],
                'description': 'Unchecked return value from low-level call',
                'recommendation': 'Always check return values from external calls'
            },
            'SC10': {
                'name': 'Short Address Attack',
                'owasp_category': 'SC10 - Short Address Attack',
                'severity': Severity.LOW,
                'patterns': [
                    r'function\s+\w+\s*\([^)]address[^)]\)\s*external',
                    r'function\s+\w+\s*\([^)]address[^)]\)\s*public',
                ],
                'description': 'Function may be vulnerable to short address attack',
                'recommendation': 'Validate input parameters length in functions accepting addresses'
            }
        }
    
    def detect_vulnerabilities(self, content: str, filename: str) -> List[Finding]:
        """Detect vulnerabilities in the given Solidity content"""
        findings = []
        parser = SolidityParser(content)
        
        # Run pattern-based detection
        findings.extend(self._pattern_based_detection(content, filename))
        
        # Run context-aware detection
        findings.extend(self._context_aware_detection(parser, filename))
        
        return findings
    
    def _pattern_based_detection(self, content: str, filename: str) -> List[Finding]:
        """Detect vulnerabilities using regex patterns"""
        findings = []
        lines = content.split('\n')
        
        for rule_id, rule in self.rules.items():
            for pattern in rule['patterns']:
                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith('//') or line.strip().startswith('/*'):
                        continue
                    
                    if re.search(pattern, line):
                        finding = Finding(
                            rule_id=rule_id,
                            title=rule['name'],
                            description=rule['description'],
                            severity=rule['severity'],
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation=rule['recommendation'],
                            owasp_category=rule['owasp_category']
                        )
                        findings.append(finding)
        
        return findings
    
    def _context_aware_detection(self, parser: SolidityParser, filename: str) -> List[Finding]:
        """Perform more sophisticated context-aware vulnerability detection"""
        findings = []
        
        # Check for missing access modifiers
        for func in parser.functions:
            if func['visibility'] in ['public', 'external'] and 'payable' in func['state_mutability']:
                # Check if there's any access control in the function
                func_content = self._get_function_body(parser.content, func['line'])
                if not re.search(r'require\s*\(|onlyOwner|onlyAdmin', func_content):
                    finding = Finding(
                        rule_id='SC04_CTX',
                        title='Payable Function Without Access Control',
                        description=f'Payable function {func["name"]} lacks access control',
                        severity=Severity.HIGH,
                        line_number=func['line'],
                        code_snippet=func['full_line'],
                        recommendation='Add proper access control to payable functions',
                        owasp_category='SC04 - Access Control Issues'
                    )
                    findings.append(finding)
        
        return findings
    
    def _get_function_body(self, content: str, start_line: int) -> str:
        """Extract function body for analysis"""
        lines = content.split('\n')
        if start_line > len(lines):
            return ""
        
        # Simple extraction - find opening brace and matching closing brace
        brace_count = 0
        function_body = []
        start_found = False
        
        for i in range(start_line - 1, len(lines)):
            line = lines[i]
            if '{' in line:
                start_found = True
            
            if start_found:
                function_body.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    break
        
        return '\n'.join(function_body)

class ReportGenerator:
    """Generate security reports in various formats"""
    
    @staticmethod
    def generate_json_report(findings: List[Finding], filename: str) -> Dict[str, Any]:
        """Generate JSON format report"""
        return {
            'file': filename,
            'timestamp': '2024-01-01T00:00:00Z',  # Would use actual timestamp
            'total_findings': len(findings),
            'findings': [
                {
                    'rule_id': f.rule_id,
                    'title': f.title,
                    'description': f.description,
                    'severity': f.severity.value,
                    'line_number': f.line_number,
                    'code_snippet': f.code_snippet,
                    'recommendation': f.recommendation,
                    'owasp_category': f.owasp_category
                }
                for f in findings
            ]
        }
    
    @staticmethod
    def generate_text_report(findings: List[Finding], filename: str) -> str:
        """Generate human-readable text report"""
        if not findings:
            return f"No vulnerabilities found in {filename}\n"
        
        report = f"\n{'='*80}\n"
        report += f"SMART CONTRACT SECURITY ANALYSIS REPORT\n"
        report += f"File: {filename}\n"
        report += f"Total Findings: {len(findings)}\n"
        report += f"{'='*80}\n\n"
        
        # Group findings by severity
        severity_groups = {}
        for finding in findings:
            severity = finding.severity.value
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(finding)
        
        # Display findings by severity
        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            if severity in severity_groups:
                report += f"\n{severity} SEVERITY FINDINGS:\n"
                report += f"{'-' * 40}\n"
                
                for finding in severity_groups[severity]:
                    report += f"\n[{finding.rule_id}] {finding.title}\n"
                    report += f"Line {finding.line_number}: {finding.code_snippet}\n"
                    report += f"Description: {finding.description}\n"
                    report += f"Category: {finding.owasp_category}\n"
                    report += f"Recommendation: {finding.recommendation}\n"
                    report += f"{'-' * 40}\n"
        
        return report

class SoliditySASTTool:
    """Main SAST tool class"""
    
    def _init_(self):
        self.detector = VulnerabilityDetector()
        self.report_generator = ReportGenerator()
    
    def scan_file(self, filepath: str) -> List[Finding]:
        """Scan a single Solidity file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.detector.detect_vulnerabilities(content, filepath)
        except Exception as e:
            print(f"Error scanning file {filepath}: {str(e)}")
            return []
    
    def scan_directory(self, directory: str) -> Dict[str, List[Finding]]:
        """Scan all .sol files in a directory"""
        results = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.sol'):
                    filepath = os.path.join(root, file)
                    findings = self.scan_file(filepath)
                    if findings:
                        results[filepath] = findings
        
        return results
    
    def generate_report(self, findings: Dict[str, List[Finding]], output_format: str = 'text') -> str:
        """Generate a comprehensive report"""
        if output_format == 'json':
            all_findings = []
            for filepath, file_findings in findings.items():
                for finding in file_findings:
                    finding_dict = self.report_generator.generate_json_report([finding], filepath)
                    all_findings.extend(finding_dict['findings'])
            
            return json.dumps({
                'total_files_scanned': len(findings),
                'total_findings': len(all_findings),
                'findings': all_findings
            }, indent=2)
        
        else:  # text format
            report = f"\n{'='*100}\n"
            report += f"SMART CONTRACT SECURITY ANALYSIS - SUMMARY REPORT\n"
            report += f"{'='*100}\n"
            report += f"Files scanned: {len(findings)}\n"
            report += f"Total findings: {sum(len(f) for f in findings.values())}\n"
            report += f"{'='*100}\n"
            
            for filepath, file_findings in findings.items():
                report += self.report_generator.generate_text_report(file_findings, filepath)
            
            return report

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Contract SAST Tool')
    parser.add_argument('target', help='File or directory to scan')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    tool = SoliditySASTTool()
    
    if os.path.isfile(args.target):
        findings = {args.target: tool.scan_file(args.target)}
    else:
        findings = tool.scan_directory(args.target)
    
    report = tool.generate_report(findings, args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)

if _name_ == '_main_':
    main()
