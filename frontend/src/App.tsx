import React from 'react';
import { Toaster } from 'react-hot-toast';
import GitHubChat from './components/GitHubChat';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-center" />
      <GitHubChat />
    </div>
  );
};

export default App; 