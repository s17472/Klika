using System;
using System.Collections;
using System.Collections.Generic;
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
            var graph = InitGraph();
            var result = new List<Vertex>();

        }

        public static int CheckAnswer(Dictionary<int, List<int>> graph)
        {


            return 1;
        }

        private static Dictionary<int, List<int>> InitGraph()
        {
            return new Dictionary<int, List<int>>()
            {
                { 0, new List<int>() { 5, 2 } },
                { 1, new List<int>() { 1, 2 } },
                { 2, new List<int>() { 1, 3, 5 }},
                { 3, new List<int>() { 4, 2 } },
                { 4, new List<int>() { 6, 5, 3 }},
                { 5, new List<int>() { 4, 1, 2 }},
                { 6, new List<int>() { 4 }}
            };
        }
    }
}
