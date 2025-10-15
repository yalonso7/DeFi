import React, { useState, useEffect } from 'react';
import { Shield, User, CheckCircle, Lock, Key, UserPlus, Users, Clock, AlertCircle } from 'lucide-react';

export default function IdentityDApp() {
  const [account, setAccount] = useState('');
  const [connected, setConnected] = useState(false);
  const [activeTab, setActiveTab] = useState('home');
  const [identity, setIdentity] = useState(null);
  const [formData, setFormData] = useState({ name: '', email: '' });
  const [attributes, setAttributes] = useState({ key: '', value: '' });
  const [accessAddress, setAccessAddress] = useState('');
  const [accessPurpose, setAccessPurpose] = useState('');
  const [accessRequests, setAccessRequests] = useState([]);
  const [notification, setNotification] = useState({ show: false, message: '', type: '' });

  // Simulated blockchain data
  const [identities, setIdentities] = useState({});
  const [userAttributes, setUserAttributes] = useState({});
  const [authorizedAccess, setAuthorizedAccess] = useState({});
  const [requests, setRequests] = useState({});

  const showNotification = (message, type = 'success') => {
    setNotification({ show: true, message, type });
    setTimeout(() => setNotification({ show: false, message: '', type: '' }), 3000);
  };

  const connectWallet = async () => {
    // Simulate wallet connection
    const mockAddress = '0x' + Math.random().toString(16).substr(2, 40);
    setAccount(mockAddress);
    setConnected(true);
    showNotification('Wallet connected successfully!');
  };

  const createIdentity = () => {
    if (!formData.name || !formData.email) {
      showNotification('Please fill all fields', 'error');
      return;
    }

    const newIdentity = {
      owner: account,
      name: formData.name,
      email: formData.email,
      isVerified: false,
      createdAt: Date.now(),
      updatedAt: Date.now()
    };

    setIdentities({ ...identities, [account]: newIdentity });
    setIdentity(newIdentity);
    setFormData({ name: '', email: '' });
    showNotification('Identity created successfully!');
  };

  const updateIdentity = () => {
    if (!identity) return;

    const updated = {
      ...identities[account],
      name: formData.name || identity.name,
      email: formData.email || identity.email,
      updatedAt: Date.now()
    };

    setIdentities({ ...identities, [account]: updated });
    setIdentity(updated);
    showNotification('Identity updated successfully!');
  };

  const addAttribute = () => {
    if (!attributes.key || !attributes.value) {
      showNotification('Please provide key and value', 'error');
      return;
    }

    const attrs = userAttributes[account] || {};
    attrs[attributes.key] = attributes.value;
    setUserAttributes({ ...userAttributes, [account]: attrs });
    setAttributes({ key: '', value: '' });
    showNotification('Attribute added successfully!');
  };

  const requestAccess = () => {
    if (!accessAddress || !accessPurpose) {
      showNotification('Please provide address and purpose', 'error');
      return;
    }

    const request = {
      requester: account,
      timestamp: Date.now(),
      approved: false,
      purpose: accessPurpose
    };

    const targetRequests = requests[accessAddress] || [];
    targetRequests.push(request);
    setRequests({ ...requests, [accessAddress]: targetRequests });
    
    setAccessAddress('');
    setAccessPurpose('');
    showNotification('Access request sent!');
  };

  const grantAccess = (requesterAddress) => {
    const access = authorizedAccess[account] || {};
    access[requesterAddress] = true;
    setAuthorizedAccess({ ...authorizedAccess, [account]: access });
    showNotification('Access granted!');
  };

  const revokeAccess = (requesterAddress) => {
    const access = authorizedAccess[account] || {};
    access[requesterAddress] = false;
    setAuthorizedAccess({ ...authorizedAccess, [account]: access });
    showNotification('Access revoked!');
  };

  useEffect(() => {
    if (account && identities[account]) {
      setIdentity(identities[account]);
      setAccessRequests(requests[account] || []);
    }
  }, [account, identities, requests]);

  const tabs = [
    { id: 'home', label: 'Home', icon: Shield },
    { id: 'identity', label: 'My Identity', icon: User },
    { id: 'attributes', label: 'Attributes', icon: Key },
    { id: 'access', label: 'Access Control', icon: Lock },
    { id: 'requests', label: 'Requests', icon: Users }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Notification */}
      {notification.show && (
        <div className={`fixed top-4 right-4 z-50 px-6 py-3 rounded-lg shadow-lg ${
          notification.type === 'error' ? 'bg-red-500' : 'bg-green-500'
        } text-white`}>
          {notification.message}
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-3">
              <Shield className="w-8 h-8 text-indigo-600" />
              <h1 className="text-2xl font-bold text-gray-900">SecureID</h1>
            </div>
            {!connected ? (
              <button
                onClick={connectWallet}
                className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
              >
                Connect Wallet
              </button>
            ) : (
              <div className="flex items-center space-x-3">
                <div className="px-4 py-2 bg-green-100 text-green-800 rounded-lg text-sm font-mono">
                  {account.substring(0, 6)}...{account.substring(38)}
                </div>
                {identity?.isVerified && (
                  <CheckCircle className="w-5 h-5 text-green-500" />
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Navigation */}
      {connected && (
        <nav className="bg-white border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center space-x-2 px-3 py-4 border-b-2 transition ${
                      activeTab === tab.id
                        ? 'border-indigo-600 text-indigo-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="font-medium">{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </nav>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!connected ? (
          <div className="text-center py-20">
            <Shield className="w-24 h-24 text-indigo-600 mx-auto mb-6" />
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Decentralized Identity Security
            </h2>
            <p className="text-lg text-gray-600 mb-8">
              Secure, private, and user-controlled digital identity on the blockchain
            </p>
            <button
              onClick={connectWallet}
              className="px-8 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition text-lg"
            >
              Get Started
            </button>
          </div>
        ) : (
          <>
            {/* Home Tab */}
            {activeTab === 'home' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-4">Dashboard</h2>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg">
                      <User className="w-8 h-8 text-indigo-600 mb-3" />
                      <p className="text-sm text-gray-600">Identity Status</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {identity ? 'Active' : 'Not Created'}
                      </p>
                    </div>
                    <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg">
                      <Key className="w-8 h-8 text-green-600 mb-3" />
                      <p className="text-sm text-gray-600">Attributes</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {Object.keys(userAttributes[account] || {}).length}
                      </p>
                    </div>
                    <div className="p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-lg">
                      <Users className="w-8 h-8 text-purple-600 mb-3" />
                      <p className="text-sm text-gray-600">Access Requests</p>
                      <p className="text-2xl font-bold text-gray-900">
                        {accessRequests.length}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4">Features</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                      <div>
                        <p className="font-semibold text-gray-900">Self-Sovereign Identity</p>
                        <p className="text-sm text-gray-600">Full control over your digital identity</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                      <div>
                        <p className="font-semibold text-gray-900">Privacy First</p>
                        <p className="text-sm text-gray-600">Granular access control to your data</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                      <div>
                        <p className="font-semibold text-gray-900">Immutable Records</p>
                        <p className="text-sm text-gray-600">Blockchain-backed security</p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <CheckCircle className="w-5 h-5 text-green-500 mt-1" />
                      <div>
                        <p className="font-semibold text-gray-900">Verifiable</p>
                        <p className="text-sm text-gray-600">Trustless verification system</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Identity Tab */}
            {activeTab === 'identity' && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">
                  {identity ? 'Update Identity' : 'Create Identity'}
                </h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name
                    </label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      placeholder={identity?.name || "Enter your name"}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address
                    </label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      placeholder={identity?.email || "Enter your email"}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    onClick={identity ? updateIdentity : createIdentity}
                    className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                  >
                    {identity ? 'Update Identity' : 'Create Identity'}
                  </button>
                </div>

                {identity && (
                  <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                    <h3 className="font-semibold text-gray-900 mb-3">Current Identity</h3>
                    <div className="space-y-2 text-sm">
                      <p><span className="font-medium">Name:</span> {identity.name}</p>
                      <p><span className="font-medium">Email:</span> {identity.email}</p>
                      <p><span className="font-medium">Status:</span> {identity.isVerified ? '✓ Verified' : '⏳ Pending'}</p>
                      <p><span className="font-medium">Created:</span> {new Date(identity.createdAt).toLocaleDateString()}</p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Attributes Tab */}
            {activeTab === 'attributes' && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Custom Attributes</h2>
                {!identity ? (
                  <div className="text-center py-8">
                    <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-3" />
                    <p className="text-gray-600">Create an identity first to add attributes</p>
                  </div>
                ) : (
                  <>
                    <div className="space-y-4 mb-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Attribute Key
                        </label>
                        <input
                          type="text"
                          value={attributes.key}
                          onChange={(e) => setAttributes({ ...attributes, key: e.target.value })}
                          placeholder="e.g., phone, country, linkedin"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Attribute Value
                        </label>
                        <input
                          type="text"
                          value={attributes.value}
                          onChange={(e) => setAttributes({ ...attributes, value: e.target.value })}
                          placeholder="Enter value"
                          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                        />
                      </div>
                      <button
                        onClick={addAttribute}
                        className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                      >
                        Add Attribute
                      </button>
                    </div>

                    <div className="space-y-2">
                      <h3 className="font-semibold text-gray-900">Stored Attributes</h3>
                      {Object.keys(userAttributes[account] || {}).length === 0 ? (
                        <p className="text-gray-500 text-sm">No attributes added yet</p>
                      ) : (
                        <div className="space-y-2">
                          {Object.entries(userAttributes[account] || {}).map(([key, value]) => (
                            <div key={key} className="p-3 bg-gray-50 rounded-lg flex justify-between items-center">
                              <div>
                                <p className="font-medium text-gray-900">{key}</p>
                                <p className="text-sm text-gray-600">{value}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Access Control Tab */}
            {activeTab === 'access' && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Access Control</h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Request Access to Address
                    </label>
                    <input
                      type="text"
                      value={accessAddress}
                      onChange={(e) => setAccessAddress(e.target.value)}
                      placeholder="0x..."
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Purpose
                    </label>
                    <textarea
                      value={accessPurpose}
                      onChange={(e) => setAccessPurpose(e.target.value)}
                      placeholder="Why do you need access?"
                      rows="3"
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    onClick={requestAccess}
                    className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition"
                  >
                    Request Access
                  </button>
                </div>
              </div>
            )}

            {/* Requests Tab */}
            {activeTab === 'requests' && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Access Requests</h2>
                {accessRequests.length === 0 ? (
                  <div className="text-center py-8">
                    <Users className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                    <p className="text-gray-600">No access requests</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {accessRequests.map((request, index) => (
                      <div key={index} className="p-4 border border-gray-200 rounded-lg">
                        <div className="flex justify-between items-start mb-3">
                          <div>
                            <p className="font-medium text-gray-900 font-mono text-sm">
                              {request.requester.substring(0, 10)}...{request.requester.substring(32)}
                            </p>
                            <p className="text-sm text-gray-600 mt-1">{request.purpose}</p>
                          </div>
                          <div className="flex items-center text-sm text-gray-500">
                            <Clock className="w-4 h-4 mr-1" />
                            {new Date(request.timestamp).toLocaleDateString()}
                          </div>
                        </div>
                        {!request.approved && (
                          <div className="flex space-x-2">
                            <button
                              onClick={() => grantAccess(request.requester)}
                              className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
                            >
                              Grant Access
                            </button>
                            <button
                              onClick={() => revokeAccess(request.requester)}
                              className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
                            >
                              Deny
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}