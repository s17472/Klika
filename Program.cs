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


    class Program
    {
        static void Main(string[] args)
        {
            // var graph = InitGraph();
            // var g2 = LoadGraph();
            // var am = new AdjacencyMatrix(g2);
            // Console.WriteLine(am);
            // SaveGraph(GenerateClique(4));
            // GradeAnswer(Checker, InitGraph());
            // GradeAnswer(Checker, LoadGraph());
            // GradeAnswer(Checker, InitClique3());
            // GradeAnswer(Checker, InitClique4());
            // GradeAnswer(Checker, GenerateClique(10));

            var xd = GetKCombs(InitGraph().Keys, 2);

            
        }

        static void Brute(Dictionary<int, List<int>> graph)
        {
            var maxPossibleCliqueSize = graph.Count;
            var maxCliqueSize = 1;
            
            for (var i = 2; i <= maxPossibleCliqueSize; i++)
            {
                var potentialCliques = GetKCombs(InitGraph().Keys, i);
                foreach (var clique in potentialCliques)
                {
                    Checker(graph.Where(x => clique.Any(x.Key) ))
                }
            }
        }
        
        public static int Checker(Dictionary<int, List<int>> graph)
        {
            var verticesIds = graph.Keys.ToArray();
            var currentScore = 0;

            foreach (var edges in graph.Values)
            {
                currentScore += edges.Count(x => verticesIds.Contains(x));
            }

            return currentScore;
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

        private static void GradeAnswer(Func<Dictionary<int, List<int>>, int> checker, Dictionary<int, List<int>> graph )
        {
            var graphsPerfectScore = GetGraphsPerfectScore(graph);
            var score = checker(graph);
            Console.WriteLine($"Perfect score: {graphsPerfectScore}");
            Console.WriteLine($"Score: {score}");
            Console.WriteLine(score == graphsPerfectScore
                ? $"Graph is a clique of { score / graph.Keys.Count + 1}"
                : $"Graph is NOT a clique");
        }

        private static int GetGraphsPerfectScore(Dictionary<int, List<int>> graph)
        {
            return graph.Keys.Count * (graph.Keys.Count - 1);
        }

        private static Dictionary<int, List<int>> InitGraph()
        {
            return new Dictionary<int, List<int>>()
            {
                { 0, new List<int>() { 4, 1 } },
                { 1, new List<int>() { 0, 2, 4 }},
                { 2, new List<int>() { 3, 1 } },
                { 3, new List<int>() { 5, 4, 3 }},
                { 4, new List<int>() { 3, 0, 1 }},
                { 5, new List<int>() { 3 }}
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
