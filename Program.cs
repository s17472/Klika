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
    }

    public class AdjacencyMatrix
    {
        private BitArray[] _matrix;
        public int Lenght { get; }

        public AdjacencyMatrix(int lenght)
        {
            _matrix = new BitArray[lenght];
            for(int i = 0; i < lenght;i++)
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
            foreach(var row in _matrix){
                foreach(bool cell in row){
                    sb.Append(cell ? 1 : 0);
                }
                sb.AppendLine();
            }

            return sb.ToString();
        }
    } 

    public class Graph
    {
        public Dictionary<Vertex, List<Vertex>> Vertices { get; set; }
        public int[] Cliques { get; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var graph = new Graph();
            var result = new List<Vertex>();

            var xd = new AdjacencyMatrix(3);
            xd.SetConnection(0,2);

            

            Console.WriteLine(xd);
        }

        public static float CheckAnswer(Graph graph, List<Vertex> answer)
        {
            return 1;
        }
    }
}
