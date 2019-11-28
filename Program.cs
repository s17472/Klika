using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Klika
{
    public class Vertex
    {
        public int Id { get; set; }
        public int X { get; set; }
        public int Y { get; set; }

        public Vertex(int id, int x, int y)
        {
            Id = id;
            X = x;
            Y = y;
        }
    }

    public class AdjacencyMatrix
    {
        private BitArray[] _matrix;
        public int Lenght { get; }

        public AdjacencyMatrix(int lenght)
        {
            _matrix = new BitArray[lenght];
            for (int i = 0; i < lenght; i++)
            {
                _matrix[i] = new BitArray(lenght);
            }

            Lenght = lenght;
        }

        public AdjacencyMatrix(Dictionary<int, List<int>> graph): this(graph.Keys.Count)
        {
            foreach (var vertex in graph)
            {
                foreach (var connectedVertecies in vertex.Value)
                {
                    SetConnection(vertex.Key, connectedVertecies);
                }
            }
        }


        public bool IsConnected(int vertex1, int vertex2)
        {
            return _matrix[vertex1][vertex2];
        }

        public void SetConnection(int vertex1, int vertex2)
        {
            _matrix[vertex1][vertex2] = _matrix[vertex2][vertex1] = true;
        }

        public void RemoveConnection(int vertex1, int vertex2)
        {
            _matrix[vertex1][vertex2] = _matrix[vertex2][vertex1] = false;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach (var row in _matrix) {
                foreach (bool cell in row) {
                    sb.Append(cell ? 1 : 0);
                }
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }

    public class SubGraph
    {
        public Dictionary<int, List<int>> Vertices { get; set; } = new Dictionary<int, List<int>>();
        public int Size => Vertices.Count;
        public double Score { get; set; } = 0.0;
        public bool IsClique => Score == 1.0;
        
        public SubGraph(Dictionary<int, List<int>> graph)
        {
            Vertices = graph;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            foreach (var vertex in Vertices)
            {
                sb.AppendLine($"{vertex.Key} : {string.Join(", ", vertex.Value)}");
            }

            sb.AppendLine($"IsClique {IsClique} : Score {Score} ");

            return sb.ToString();
        }
    }
    

    public class Program
    {
        static void Main(string[] args)
        {
            // var g2 = LoadGraph();
            // var am = new AdjacencyMatrix(g2);
            // Console.WriteLine(am);
            // SaveGraph(GenerateClique(4));
            // GradeAnswer(Checker, InitGraph());
            // GradeAnswer(Checker, LoadGraph());
            // GradeAnswer(Checker, InitClique3());
            // GradeAnswer(Checker, InitClique4());
            // GradeAnswer(Checker, GenerateClique(10));

            var graph = InitGraph();
            var subGraphs = GeneratePotentialCliques(graph);
            Brute(subGraphs);
            ReviewRecord(subGraphs);
        }

        static SubGraph NextSubGraph(Dictionary<int, List<int>> graph, SubGraph subGraph)
        {
            
        }
        
        static List<SubGraph> GeneratePotentialCliques(Dictionary<int, List<int>> graph)
        {
            var subGraphs = new List<SubGraph>();

            for (var i = 2; i <= graph.Count; i++)
            {
                var potentialCliques = GetKCombs(graph.Keys, i);
                foreach (var clique in potentialCliques)
                {
                    var subGraph = new SubGraph(graph.Where(x => clique.Contains(x.Key))
                        .ToDictionary(x => x.Key, x => x.Value));
                    subGraphs.Add(subGraph);
                }
            }

            return subGraphs;
        }
        
        private static void Brute(List<SubGraph> subGraphs)
        {
            foreach (var potentialClique in subGraphs)
            {
                potentialClique.Score = Checker(potentialClique.Vertices);
            }
        }
        
        private static void ReviewRecord(List<SubGraph> subGraphs)
        {
            Console.WriteLine("All cliques in graph: ");
            subGraphs.Where(x => x.IsClique).ToList().ForEach(Console.WriteLine);
            var largestClique = subGraphs.OrderByDescending(x => x.Size).FirstOrDefault(x => x.IsClique);
            Console.WriteLine($"Largest clique size: {largestClique.Size}");
            Console.WriteLine($"Largest clique: {string.Join(", ", largestClique.Vertices.Keys)}");
        }
        
        //  kod greya
        // lista sasiadów
        
        
        private static double Checker(Dictionary<int, List<int>> graph)
        {
            var verticesIds = graph.Keys.ToArray();
            var currentScore = 0.0;

            foreach (var edges in graph.Values)
            {
                currentScore += edges.Count(x => verticesIds.Contains(x));
            }

            return currentScore / ((graph.Count - 1) * graph.Count);
        }

        private static Dictionary<int, List<int>> LoadGraph(string file = "graph.txt")
        {
            var graph = new Dictionary<int, List<int>>();

            using (var sr = new StreamReader(file))
            {
                var line = "";
                while ((line = sr.ReadLine()) != null)
                {
                    var s = line.Trim().Split(" ");

                    if (graph.TryGetValue(Int32.Parse(s[0]), out var edges))
                    {
                        edges.Add(Int32.Parse(s[1]));
                    }
                    else
                    {
                        graph.Add(Int32.Parse(s[0]), new List<int>() { Int32.Parse(s[1]) });
                    }
                }
            }

            return graph;
        }

        private static void SaveGraph(Dictionary<int, List<int>> graph, string file = "answer.txt")
        {
            using (var sw = new StreamWriter(file))
            {
                foreach (var vertex in graph)
                {
                    foreach (var edge in vertex.Value)
                    {
                        sw.WriteLine($"{vertex.Key} {edge}");   
                    }
                }

            }
        }
        
        static IEnumerable<IEnumerable<T>> GetKCombs<T>(IEnumerable<T> list, int length) where T : IComparable
        {
            if (length == 1) return list.Select(t => new T[] {t});
            return  GetKCombs(list, length - 1)
                .SelectMany(t => list.Where(o => o.CompareTo(t.Last()) > 0),
                    (t1, t2) => t1.Concat(new T[] {t2}));
        }

        private static Dictionary<int, List<int>> GenerateClique(int n)
        {
            var grapgh = new Dictionary<int, List<int>>(n);
            var r = Enumerable.Range(0, n);

            for (int i = 0; i < n; i++)
            {
                grapgh.Add(i, r.Where(x => x != i).ToList());
            }

            return grapgh;
        }
        
        private static int GetGraphsPerfectScore(Dictionary<int, List<int>> graph)
        {
            return graph.Keys.Count * (graph.Keys.Count - 1);
        }

        private static Dictionary<int, List<int>> InitGraph()
        {
            return new Dictionary<int, List<int>>()
            {
                { 0, new List<int>() { 4, 1, 6 } },
                { 1, new List<int>() { 0, 2, 4, 6 }},
                { 2, new List<int>() { 3, 1 } },
                { 3, new List<int>() { 5, 4, 2 }},
                { 4, new List<int>() { 3, 0, 1, 6 }},
                { 5, new List<int>() { 3 }},
                { 6, new List<int>() { 0, 1, 4 }}
            };
        }

        private static Dictionary<int, List<int>> InitClique3()
        {
            return new Dictionary<int, List<int>>()
            {
                { 0, new List<int>() { 4, 1, 14,7,5,3} },
                { 1, new List<int>() { 0, 2, 4 }},
                { 4, new List<int>() { 3, 0, 1 }},
            };
        }

        private static Dictionary<int, List<int>> InitClique4()
        {
            return new Dictionary<int, List<int>>()
            {
                { 0, new List<int>() { 4, 1, 2 } },
                { 1, new List<int>() { 0, 2, 4 }},
                { 4, new List<int>() { 3, 0, 1, 2, 10, 12}},
                { 2, new List<int>() { 4, 0, 1 }},
            };
        }
    }
}