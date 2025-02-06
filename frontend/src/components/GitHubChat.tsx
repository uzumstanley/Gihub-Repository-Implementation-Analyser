import React, { useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { Loader2, ChevronDown, ChevronRight, Github } from "lucide-react";
import { toast } from "react-hot-toast";

interface DocumentMetadata {
  file_path: string;
  type: string;
  is_code: boolean;
  is_implementation: boolean;
  title: string;
}

interface Document {
  text: string;
  meta_data: DocumentMetadata;
}

interface QueryResponse {
  rationale: string;
  answer: string;
  contexts: Document[];
}

const GitHubChat: React.FC = () => {
  const [repoUrl, setRepoUrl] = useState("");
  const [query, setQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [expandedContexts, setExpandedContexts] = useState<{
    [key: number]: boolean;
  }>({});

  const analyzeRepo = useCallback(async () => {
    if (!repoUrl.trim() || !query.trim()) return;

    setIsProcessing(true);
    try {
      const response = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          repo_url: repoUrl,
          query: query,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to analyze repository");
      }

      const result = await response.json();
      setResponse(result);
      toast.success("Analysis complete!");
    } catch (error) {
      console.error("Error:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Failed to analyze repository";
      toast.error(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  }, [repoUrl, query]);

  const toggleContext = (index: number) => {
    setExpandedContexts((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-4xl mx-auto px-4 py-12 space-y-8">
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2">
            <Github className="h-8 w-8" />
            <h1 className="text-3xl font-bold tracking-tight">GitHubChat</h1>
          </div>
          <p className="text-gray-600">Chat with any Github Repo!</p>
        </div>

        <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl p-6 space-y-6">
          <div className="space-y-4">
            <div>
              <label
                htmlFor="repo-url"
                className="block text-sm font-medium text-gray-700"
              >
                GitHub Repository URL
              </label>
              <div className="mt-1">
                <input
                  id="repo-url"
                  type="text"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-black focus:ring-black sm:text-sm px-4 py-2 border"
                  placeholder="https://github.com/username/repository"
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="query"
                className="block text-sm font-medium text-gray-700"
              >
                Query
              </label>
              <div className="mt-1">
                <textarea
                  id="query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={3}
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-black focus:ring-black sm:text-sm px-4 py-2 border"
                  placeholder="What would you like to know about this repository?"
                />
              </div>
            </div>

            <button
              onClick={analyzeRepo}
              disabled={!repoUrl.trim() || !query.trim() || isProcessing}
              className="w-full flex items-center justify-center gap-2 rounded-md bg-black px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-black/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-black disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <span>Analyze Repository</span>
              )}
            </button>
          </div>
        </div>

        {response && (
          <div className="bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl divide-y divide-gray-200">
            <div className="p-6 space-y-4">
              <h2 className="text-lg font-semibold">Analysis</h2>
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{response.rationale}</ReactMarkdown>
              </div>
            </div>

            <div className="p-6 space-y-4">
              <h2 className="text-lg font-semibold">Answer</h2>
              <div className="prose prose-sm max-w-none">
                <ReactMarkdown>{response.answer}</ReactMarkdown>
              </div>
            </div>

            <div className="p-6 space-y-4">
              <h2 className="text-lg font-semibold">Context</h2>
              <div className="space-y-3">
                {response.contexts.map((context, index) => (
                  <div
                    key={index}
                    className="border rounded-lg overflow-hidden bg-gray-50"
                  >
                    <button
                      onClick={() => toggleContext(index)}
                      className="w-full flex items-center gap-2 p-4 text-left hover:bg-gray-100 transition-colors"
                    >
                      {expandedContexts[index] ? (
                        <ChevronDown className="h-4 w-4 flex-shrink-0 text-gray-500" />
                      ) : (
                        <ChevronRight className="h-4 w-4 flex-shrink-0 text-gray-500" />
                      )}
                      <span className="text-sm font-medium text-gray-900 truncate">
                        {context.meta_data.file_path}
                      </span>
                    </button>
                    {expandedContexts[index] && (
                      <div className="px-4 pb-4 border-t bg-white">
                        <div className="prose prose-sm max-w-none mt-4">
                          <ReactMarkdown>{context.text}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="p-6">
              <button
                onClick={() => {
                  setResponse(null);
                  setQuery("");
                }}
                className="w-full flex items-center justify-center rounded-md bg-gray-50 px-4 py-2.5 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-100"
              >
                Ask Another Question
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GitHubChat;
