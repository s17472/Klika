using System;
using System.Collections.Generic;

namespace Klika
{
    public class Vertex
    {
        public int Id { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var graph = new Dictionary<Vertex, List<Vertex>>();
            var result = new List<Vertex>();


            Console.WriteLine("Hello World!");
        }

        public static float CheckAnswer(Dictionary<Vertex, List<Vertex>> graph, List<Vertex> answer)
        {
            return 1;
        }
    }
}
