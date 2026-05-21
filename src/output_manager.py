# Output Directory Manager: Handles organized file output with timestamped directories
import os
from datetime import datetime
from pathlib import Path


class OutputManager:
    """Manages organized output directories for the research assistant"""
    
    def __init__(self, base_dir="outputs"):
        self.base_dir = base_dir
        self.session_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create main output directories
        self.dirs = {
            'bibtex': os.path.join(base_dir, 'bibtex'),
            'searches': os.path.join(base_dir, 'saved_searches'),
            'evaluations': os.path.join(base_dir, 'evaluations'),
            'logs': os.path.join(base_dir, 'logs')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def get_bibtex_path(self, filename: str, use_timestamp: bool = True) -> str:
        """
        Get organized path for BibTeX export
        
        Args:
            filename: Desired filename (e.g., 'transformers.bib')
            use_timestamp: If True, creates timestamped subfolder
        
        Returns:
            Full path where file should be saved
        """
        if use_timestamp:
            # Create session folder: outputs/bibtex/2025-10-24_07-14-21/
            session_dir = os.path.join(self.dirs['bibtex'], self.session_timestamp)
            os.makedirs(session_dir, exist_ok=True)
            return os.path.join(session_dir, filename)
        else:
            # Save directly in bibtex folder
            return os.path.join(self.dirs['bibtex'], filename)
    
    def get_search_path(self, filename: str) -> str:
        """Get path for saved search results"""
        session_dir = os.path.join(self.dirs['searches'], self.session_timestamp)
        os.makedirs(session_dir, exist_ok=True)
        return os.path.join(session_dir, filename)
    
    def get_evaluation_path(self, filename: str) -> str:
        """Get path for evaluation results"""
        session_dir = os.path.join(self.dirs['evaluations'], self.session_timestamp)
        os.makedirs(session_dir, exist_ok=True)
        return os.path.join(session_dir, filename)
    
    def get_session_summary(self) -> dict:
        """Get summary of current session's output"""
        session_dir = os.path.join(self.dirs['bibtex'], self.session_timestamp)
        
        if not os.path.exists(session_dir):
            return {'session': self.session_timestamp, 'files': 0}
        
        files = [f for f in os.listdir(session_dir) if f.endswith('.bib')]
        
        return {
            'session': self.session_timestamp,
            'directory': session_dir,
            'files': len(files),
            'file_list': files
        }
    
    def create_latest_link(self):
        """Create 'latest' symlink to most recent session (Windows compatible)"""
        latest_path = os.path.join(self.dirs['bibtex'], 'latest.txt')
        session_dir = os.path.join(self.dirs['bibtex'], self.session_timestamp)
        
        # Write path to latest.txt (Windows-compatible alternative to symlink)
        with open(latest_path, 'w') as f:
            f.write(f"Latest session: {self.session_timestamp}\n")
            f.write(f"Directory: {session_dir}\n")
    
    def list_all_sessions(self) -> list:
        """List all BibTeX export sessions"""
        bibtex_dir = self.dirs['bibtex']
        
        sessions = []
        for item in os.listdir(bibtex_dir):
            item_path = os.path.join(bibtex_dir, item)
            if os.path.isdir(item_path) and item != 'latest':
                # Count .bib files in session
                bib_files = [f for f in os.listdir(item_path) if f.endswith('.bib')]
                sessions.append({
                    'timestamp': item,
                    'path': item_path,
                    'file_count': len(bib_files)
                })
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        return sessions
    
    def print_session_info(self):
        """Print information about current session"""
        print("\n" + "="*70)
        print("OUTPUT ORGANIZATION")
        print("="*70)
        print(f"\nSession: {self.session_timestamp}")
        print(f"\nOutput directories:")
        print(f"  BibTeX exports: {self.dirs['bibtex']}/{self.session_timestamp}/")
        print(f"  Saved searches: {self.dirs['searches']}/{self.session_timestamp}/")
        print(f"  Evaluations:    {self.dirs['evaluations']}/{self.session_timestamp}/")
        print("="*70 + "\n")


# Global output manager instance
_output_manager = None

def get_output_manager() -> OutputManager:
    """Get or create global output manager"""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager


if __name__ == "__main__":
    # Test the output manager
    manager = OutputManager()
    
    print("Output Manager Test")
    print("=" * 70)
    
    # Show structure
    manager.print_session_info()
    
    # Test paths
    print("\nExample paths:")
    print(f"BibTeX:     {manager.get_bibtex_path('transformers.bib')}")
    print(f"Search:     {manager.get_search_path('search_results.txt')}")
    print(f"Evaluation: {manager.get_evaluation_path('test_results.json')}")
    
    # Show sessions
    print("\n" + "=" * 70)
    print("Previous sessions:")
    sessions = manager.list_all_sessions()
    if sessions:
        for session in sessions[:5]:  # Show last 5
            print(f"  {session['timestamp']}: {session['file_count']} files")
    else:
        print("  No previous sessions")
    
    print("\n✓ Output manager ready!")