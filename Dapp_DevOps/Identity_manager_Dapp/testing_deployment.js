// ===================================================================
// COMPREHENSIVE TESTING SUITE
// ===================================================================

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time, loadFixture } = require("@nomicfoundation/hardhat-network-helpers");

describe("DecentralizedIdentity - Complete Test Suite", function () {
  
  // Test fixture for deployment
  async function deployIdentityFixture() {
    const [admin, user1, user2, user3, verifier] = await ethers.getSigners();
    
    const DecentralizedIdentity = await ethers.getContractFactory("DecentralizedIdentity");
    const identity = await DecentralizedIdentity.deploy();
    await identity.waitForDeployment();
    
    return { identity, admin, user1, user2, user3, verifier };
  }

  describe("Deployment", function () {
    it("Should set the correct admin", async function () {
      const { identity, admin } = await loadFixture(deployIdentityFixture);
      expect(await identity.admin()).to.equal(admin.address);
    });

    it("Should initialize with zero identities", async function () {
      const { identity } = await loadFixture(deployIdentityFixture);
      expect(await identity.totalIdentities()).to.equal(0);
    });
  });

  describe("Identity Management", function () {
    describe("Creation", function () {
      it("Should create identity with valid data", async function () {
        const { identity, user1 } = await loadFixture(deployIdentityFixture);
        
        await expect(identity.connect(user1).createIdentity("Alice", "alice@test.com"))
          .to.emit(identity, "IdentityCreated")
          .withArgs(user1.address, await time.latest());
        
        expect(await identity.hasIdentity(user1.address)).to.be.true;
        expect(await identity.totalIdentities()).to.equal(1);
      });

      it("Should reject empty name", async function () {
        const { identity, user1 } = await loadFixture(deployIdentityFixture);
        
        await expect(
          identity.connect(user1).createIdentity("", "alice@test.com")
        ).to.be.revertedWith("Name required");
      });

      it("Should prevent duplicate creation", async function () {
        const { identity, user1 } = await loadFixture(deployIdentityFixture);
        
        await identity.connect(user1).createIdentity("Alice", "alice@test.com");
        
        await expect(
          identity.connect(user1).createIdentity("Bob", "bob@test.com")
        ).to.be.revertedWith("Identity already exists");
      });

      it("Should store correct identity data", async function () {
        const { identity, user1 } = await loadFixture(deployIdentityFixture);
        
        await identity.connect(user1).createIdentity("Alice", "alice@test.com");
        
        const [owner, name, email, isVerified, createdAt, updatedAt] = 
          await identity.connect(user1).getIdentity(user1.address);
        
        expect(owner).to.equal(user1.address);
        expect(name).to.equal("Alice");
        expect(email).to.equal("alice@test.com");
        expect(isVerified).to.be.false;
        expect(createdAt).to.be.gt(0);
        expect(updatedAt).to.equal(createdAt);
      });
    });

    describe("Updates", function () {
      it("Should update identity information", async function () {
        const { identity, user1 } = await loadFixture(deployIdentityFixture);
        
        await identity.connect(user1).createIdentity("Alice", "alice@test.com");
        
        await expect(
          identity.connect(user1).updateIdentity("Alice Smith", "alice.smith@test.com")
        ).to.emit(identity, "IdentityUpdated");
        
        const [, name, email] = await identity.connect(user1).getIdentity(user1.address);
        expect(name).to.equal("Alice Smith");
        expect(email).to.equal("alice.smith@test.com");
      });

      it("Should update timestamp on update", async function () {
        const { identity, user1 } = await loadFixture(deployIdentityFixture);
        
        await identity.connect(user1).createIdentity("Alice", "alice@test.com");
        const [,,,,, updatedAt1] = await identity.connect(user1).getIdentity(user1.address);
        
        await time.increase(3600); // 1 hour
        
        await identity.connect(user1).updateIdentity("Alice Smith", "alice@test.com");
        const [,,,,, updatedAt2] = await identity.connect(user1).getIdentity(user1.address);
        
        expect(updatedAt2).to.be.gt(updatedAt1);
      });

      it("Should reject updates from non-owners", async function () {
        const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
        
        await identity.connect(user1).createIdentity("Alice", "alice@test.com");
        
        await expect(
          identity.connect(user2).updateIdentity("Hacker", "hack@test.com")
        ).to.be.revertedWith("No identity found");
      });
    });
  });

  describe("Attributes", function () {
    it("Should add custom attributes", async function () {
      const { identity, user1 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await expect(identity.connect(user1).addAttribute("phone", "+1234567890"))
        .to.emit(identity, "AttributeAdded")
        .withArgs(user1.address, "phone");
      
      const phone = await identity.connect(user1).getAttribute(user1.address, "phone");
      expect(phone).to.equal("+1234567890");
    });

    it("Should add multiple attributes", async function () {
      const { identity, user1 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      await identity.connect(user1).addAttribute("phone", "+1234567890");
      await identity.connect(user1).addAttribute("country", "USA");
      await identity.connect(user1).addAttribute("github", "alice123");
      
      expect(await identity.connect(user1).getAttribute(user1.address, "phone"))
        .to.equal("+1234567890");
      expect(await identity.connect(user1).getAttribute(user1.address, "country"))
        .to.equal("USA");
      expect(await identity.connect(user1).getAttribute(user1.address, "github"))
        .to.equal("alice123");
    });

    it("Should update existing attributes", async function () {
      const { identity, user1 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      await identity.connect(user1).addAttribute("phone", "+1234567890");
      await identity.connect(user1).addAttribute("phone", "+0987654321");
      
      const phone = await identity.connect(user1).getAttribute(user1.address, "phone");
      expect(phone).to.equal("+0987654321");
    });

    it("Should restrict attribute access to authorized users", async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      await identity.connect(user1).addAttribute("ssn", "123-45-6789");
      
      await expect(
        identity.connect(user2).getAttribute(user1.address, "ssn")
      ).to.be.revertedWith("Access denied");
    });

    it("Should allow admin to access all attributes", async function () {
      const { identity, admin, user1 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      await identity.connect(user1).addAttribute("private", "secret");
      
      const value = await identity.connect(admin).getAttribute(user1.address, "private");
      expect(value).to.equal("secret");
    });
  });

  describe("Access Control", function () {
    beforeEach(async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      await identity.connect(user2).createIdentity("Bob", "bob@test.com");
    });

    it("Should request access", async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      await identity.connect(user2).createIdentity("Bob", "bob@test.com");
      
      await expect(
        identity.connect(user2).requestAccess(user1.address, "Need verification")
      ).to.emit(identity, "AccessRequested")
        .withArgs(user2.address, user1.address);
    });

    it("Should grant access", async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await expect(identity.connect(user1).grantAccess(user2.address))
        .to.emit(identity, "AccessGranted")
        .withArgs(user1.address, user2.address);
      
      expect(await identity.hasAccess(user1.address, user2.address)).to.be.true;
    });

    it("Should revoke access", async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await identity.connect(user1).grantAccess(user2.address);
      
      await expect(identity.connect(user1).revokeAccess(user2.address))
        .to.emit(identity, "AccessRevoked")
        .withArgs(user1.address, user2.address);
      
      expect(await identity.hasAccess(user1.address, user2.address)).to.be.false;
    });

    it("Should allow accessing attributes after grant", async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await identity.connect(user1).addAttribute("phone", "+1234567890");
      await identity.connect(user1).grantAccess(user2.address);
      
      const phone = await identity.connect(user2).getAttribute(user1.address, "phone");
      expect(phone).to.equal("+1234567890");
    });

    it("Should prevent self-access request", async function () {
      const { identity, user1 } = await loadFixture(deployIdentityFixture);
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await expect(
        identity.connect(user1).requestAccess(user1.address, "Self request")
      ).to.be.revertedWith("Cannot request access to own identity");
    });
  });

  describe("Verification", function () {
    it("Should verify identity", async function () {
      const { identity, admin, user1 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await expect(identity.connect(admin).verifyIdentity(user1.address))
        .to.emit(identity, "IdentityVerified")
        .withArgs(user1.address, admin.address);
      
      expect(await identity.isVerified(user1.address)).to.be.true;
    });

    it("Should reject verification from non-admin", async function () {
      const { identity, user1, user2 } = await loadFixture(deployIdentityFixture);
      
      await identity.connect(user1).createIdentity("Alice", "alice@test.com");
      
      await expect(
        identity.connect(user2).verifyIdentity(user1.address)
      ).to.be.revertedWith("Only admin can verify");
    });

    it("Should reject verification of non-existent identity", async function () {
      const { identity, admin, user1 } = await loadFixture(deployIdentityFixture);
      
      await expect(
        identity.connect(admin).verifyIdentity(user1.address)
      ).to.be.revertedWith("Identity does not exist");
    });
  });

  describe("Gas Optimization Tests", function () {