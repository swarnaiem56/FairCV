import pandas as pd
import networkx as nx
import json
from pathlib import Path


class HiringGraphBuilder:
    """Build a real hiring network graph from resume dataset."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.G = nx.DiGraph()
        
    def load_dataset(self):
        """Load resume dataset."""
        print(f"Loading dataset from {self.dataset_path}...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Loaded {len(self.df)} candidates")
        return self.df
    
    def build_graph(self):
        """Build heterogeneous hiring graph with nodes and edges."""
        if self.df is None:
            self.load_dataset()
        
        print("\nBuilding hiring graph...")
        
        # 1. Add candidate nodes
        print("  Adding candidate nodes...")
        for idx, row in self.df.iterrows():
            cand_id = row['candidate_id']
            self.G.add_node(
                cand_id,
                type='candidate',
                name=row['full_name'],
                gender=row['gender'],
                location=row['location'],
                years_experience=row['years_experience'],
                technical_score=row['technical_score'],
                communication_score=row['communication_score'],
                aptitude_score=row['aptitude_score'],
                resume_score=row['resume_score'],
                hiring_score=row['hiring_score'],
                selected=row['selected']
            )
        
        # 2. Add university nodes and edges
        print("  Adding university nodes...")
        universities = self.df[['university_name', 'university_tier', 'university_type', 'university_region']].drop_duplicates()
        for idx, row in universities.iterrows():
            if pd.isna(row['university_name']):
                continue
            uni_id = f"UNI_{row['university_name'].replace(' ', '_')}"
            self.G.add_node(
                uni_id,
                type='university',
                name=row['university_name'],
                tier=row['university_tier'],
                uni_type=row['university_type'],
                region=row['university_region']
            )
        
        # Add attended edges
        print("  Adding attended (candidate-university) edges...")
        for idx, row in self.df.iterrows():
            cand_id = row['candidate_id']
            university_name = row['university_name']
            if pd.isna(university_name):
                continue
            uni_id = f"UNI_{university_name.replace(' ', '_')}"
            self.G.add_edge(
                cand_id,
                uni_id,
                relation='attended',
                degree=row['degree'],
                specialization=row['specialization'],
                graduation_year=row['graduation_year'],
                cgpa=row['cgpa']
            )
        
        # 3. Add skill nodes and edges
        print("  Adding skill nodes...")
        all_skills = set()
        for skills_str in self.df['skills'].dropna():
            if isinstance(skills_str, str):
                skills = [s.strip() for s in skills_str.split('|') if s.strip()]
                all_skills.update(skills)
        
        skill_nodes = {f"SKILL_{skill.lower().replace(' ', '_')}" for skill in all_skills}
        for skill_id in skill_nodes:
            skill_name = skill_id.replace('SKILL_', '').replace('_', ' ')
            self.G.add_node(skill_id, type='skill', name=skill_name)
        
        # Add has_skill edges
        print("  Adding skill edges (candidate-skill)...")
        for idx, row in self.df.iterrows():
            cand_id = row['candidate_id']
            skills_str = row['skills']
            if isinstance(skills_str, str):
                skills = [s.strip() for s in skills_str.split('|') if s.strip()]
                for skill in skills:
                    skill_id = f"SKILL_{skill.lower().replace(' ', '_')}"
                    if skill_id in skill_nodes:
                        self.G.add_edge(cand_id, skill_id, relation='has_skill')
        
        # 4. Add location nodes and edges
        print("  Adding location nodes...")
        locations = sorted(self.df['candidate_region'].dropna().unique())
        for loc in locations:
            if pd.notna(loc):
                loc_id = f"LOC_{loc.lower().replace(' ', '_')}"
                self.G.add_node(loc_id, type='location', name=loc)
        
        # Add located_at edges
        print("  Adding location edges (candidate-region)...")
        for idx, row in self.df.iterrows():
            cand_id = row['candidate_id']
            loc = row['candidate_region']
            if pd.notna(loc):
                loc_id = f"LOC_{loc.lower().replace(' ', '_')}"
                self.G.add_edge(cand_id, loc_id, relation='located_at')
        
        # 5. Infer company nodes from hiring score (binary classification)
        print("  Adding company nodes...")
        median_score = self.df['hiring_score'].median()
        companies = ["Company_A", "Company_B", "Company_C", "Company_D", "Company_E"]
        for comp in companies:
            self.G.add_node(comp, type='company')
        
        # Add hired edges based on deterministic assignment
        print("  Adding hiring edges (company-candidate)...")
        for idx, row in self.df.iterrows():
            cand_id = row['candidate_id']
            hiring_score = row['hiring_score']
            years_exp = row['years_experience']
            uni_tier = row['university_tier']
            
            # Assign candidates to companies based on heuristics
            if hiring_score > median_score * 1.1:
                self.G.add_edge("Company_A", cand_id, relation='hired', score=hiring_score)
            
            if years_exp > 3:
                self.G.add_edge("Company_B", cand_id, relation='hired', score=hiring_score)
            
            if uni_tier == 'Tier 1':
                self.G.add_edge("Company_C", cand_id, relation='hired', score=hiring_score)
            
            if row['technical_score'] > 7 and row['communication_score'] > 7:
                self.G.add_edge("Company_D", cand_id, relation='hired', score=hiring_score)
            
            if row['selected'] == 'Yes':
                self.G.add_edge("Company_E", cand_id, relation='hired', score=hiring_score)
        
        print(f"\nGraph built successfully!")
        print(f"  Total nodes: {self.G.number_of_nodes()}")
        print(f"  Total edges: {self.G.number_of_edges()}")
        return self.G
    
    def save_graph(self, output_path="hiring_graph.graphml"):
        """Save graph to GraphML format."""
        try:
            nx.write_graphml(self.G, output_path)
            print(f"\nGraph saved to {output_path}")
        except Exception as e:
            print(f"Failed to save GraphML: {e}")
    
    def print_summary(self):
        """Print node and edge statistics."""
        print("\n" + "="*60)
        print("HIRING GRAPH SUMMARY")
        print("="*60)
        
        node_types = {}
        for n, d in self.G.nodes(data=True):
            t = d.get('type', 'unknown')
            node_types.setdefault(t, 0)
            node_types[t] += 1
        
        print("\nNode Types:")
        for t, cnt in sorted(node_types.items()):
            print(f"  {t:15s}: {cnt}")
        print(f"  {'TOTAL':15s}: {self.G.number_of_nodes()}")
        
        edge_types = {}
        for u, v, d in self.G.edges(data=True):
            rel = d.get('relation', 'unknown')
            edge_types.setdefault(rel, 0)
            edge_types[rel] += 1
        
        print("\nEdge Types:")
        for rel, cnt in sorted(edge_types.items()):
            print(f"  {rel:20s}: {cnt}")
        print(f"  {'TOTAL':20s}: {self.G.number_of_edges()}")
        
        print("\nHiring Statistics:")
        hired_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get('relation') == 'hired']
        print(f"  Total hires: {len(hired_edges)}")
        companies_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'company']
        for comp in companies_nodes:
            hired_by_comp = sum(1 for u, v in hired_edges if u == comp)
            print(f"    {comp}: {hired_by_comp}")
        
        print("\n" + "="*60)
    
    def export_statistics(self, output_file="graph_stats.json"):
        """Export graph statistics to JSON."""
        stats = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "node_types": {},
            "edge_types": {},
            "candidates": {},
            "companies": {}
        }
        
        # Count node types
        for n, d in self.G.nodes(data=True):
            t = d.get('type', 'unknown')
            stats['node_types'][t] = stats['node_types'].get(t, 0) + 1
        
        # Count edge types
        for u, v, d in self.G.edges(data=True):
            rel = d.get('relation', 'unknown')
            stats['edge_types'][rel] = stats['edge_types'].get(rel, 0) + 1
        
        # Candidate statistics
        candidates_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'candidate']
        stats['candidates']['total'] = len(candidates_nodes)
        
        hired_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get('relation') == 'hired']
        hired_cands = set(v for u, v in hired_edges)
        stats['candidates']['hired'] = len(hired_cands)
        
        # Company statistics
        for comp in [n for n, d in self.G.nodes(data=True) if d.get('type') == 'company']:
            hired_by_comp = sum(1 for u, v in hired_edges if u == comp)
            stats['companies'][comp] = hired_by_comp
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics exported to {output_file}")


if __name__ == "__main__":
    dataset_path = r'C:\Users\LOQ\OneDrive\Datasets\resume_dataset_augmented.csv'
    
    builder = HiringGraphBuilder(dataset_path)
    builder.load_dataset()
    builder.build_graph()
    builder.print_summary()
    
    builder.save_graph('hiring_graph.graphml')
    builder.export_statistics('hiring_graph_stats.json')
    
    print("\nDone! Generated files:")
    print("  - hiring_graph.graphml (graph structure)")
    print("  - hiring_graph_stats.json (statistics)")
