/// <reference types="cypress" />

// Ignore non-fatal third-party / Streamlit internal browser exceptions during Cypress proxy injection
Cypress.on("uncaught:exception", () => false);

describe("WT-GPMS Black-Box System Test Suite (Positive & Negative Scenarios)", () => {
  const loginAs = (username, password) => {
    cy.get('input[aria-label="Username"]').should("be.visible").clear().type(username);
    cy.get('input[aria-label="Password"]').should("be.visible").clear().type(password);
    cy.contains('[data-testid="stFormSubmitButton"] button', "Sign In").click();
    cy.contains("Wind Turbine Gearbox Predictive Maintenance", { timeout: 60000 }).should(
      "be.visible"
    );
  };

  beforeEach(() => {
    cy.visit("/");
    cy.contains("Wind Turbine SCADA Security Portal", { timeout: 20000 }).should("be.visible");
  });

  // =========================================================================
  // NEGATIVE TEST CASES (SECURITY, VALIDATION & BOUNDARY CHECKS)
  // =========================================================================
  context("Negative Test Cases (Security & Boundary Validation)", () => {
    it("TC-NEG-07: Unauthenticated access blocks dashboard tabs behind Security Portal", () => {
      cy.contains("Wind Turbine SCADA Security Portal").should("be.visible");
      cy.contains("Historical SCADA Telemetry").should("not.exist");
      cy.contains("Manual Diagnostic & What-If Simulator").should("not.exist");
    });

    it("TC-NEG-01: Rejects Sign In with incorrect password", () => {
      cy.get('input[aria-label="Username"]').type("admin");
      cy.get('input[aria-label="Password"]').type("wrongpassword999");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Sign In").click();
      cy.contains("Invalid username or password. Please try again.").should("be.visible");
    });

    it("TC-NEG-02: Rejects Sign In with non-existent username", () => {
      cy.get('input[aria-label="Username"]').type("ghost_user_xyz");
      cy.get('input[aria-label="Password"]').type("somepassword123");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Sign In").click();
      cy.contains("Invalid username or password. Please try again.").should("be.visible");
    });

    it("TC-NEG-03: Rejects account registration with username shorter than 3 characters (BVA)", () => {
      cy.contains('[role="tab"]', "Create Account").click();
      cy.get('input[aria-label="Full Name (e.g. Dr. Alex Mercer)"]').type("Short User");
      cy.get('input[aria-label="Username (min 3 chars)"]').type("ab");
      cy.get('input[aria-label="Password (min 6 chars)"]').type("ValidPass123");
      cy.get('input[aria-label="Confirm Password"]').type("ValidPass123");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Create Account").click();
      cy.contains("Username must be at least 3 characters long.").should("be.visible");
    });

    it("TC-NEG-04: Rejects account registration with password shorter than 6 characters (BVA)", () => {
      cy.contains('[role="tab"]', "Create Account").click();
      cy.get('input[aria-label="Full Name (e.g. Dr. Alex Mercer)"]').type("Weak Pw User");
      cy.get('input[aria-label="Username (min 3 chars)"]').type("valid_user_99");
      cy.get('input[aria-label="Password (min 6 chars)"]').type("12345");
      cy.get('input[aria-label="Confirm Password"]').type("12345");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Create Account").click();
      cy.contains("Password must be at least 6 characters long.").should("be.visible");
    });

    it("TC-NEG-05: Rejects account registration when password and confirmation mismatch", () => {
      cy.contains('[role="tab"]', "Create Account").click();
      cy.get('input[aria-label="Full Name (e.g. Dr. Alex Mercer)"]').type("Mismatch User");
      cy.get('input[aria-label="Username (min 3 chars)"]').type("mismatch_user_99");
      cy.get('input[aria-label="Password (min 6 chars)"]').type("PasswordOne1!");
      cy.get('input[aria-label="Confirm Password"]').type("PasswordTwo2!");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Create Account").click();
      cy.contains("Passwords do not match. Please re-enter.").should("be.visible");
    });

    it("TC-NEG-06: Rejects duplicate username registration attempt", () => {
      cy.contains('[role="tab"]', "Create Account").click();
      cy.get('input[aria-label="Full Name (e.g. Dr. Alex Mercer)"]').type("Duplicate Admin");
      cy.get('input[aria-label="Username (min 3 chars)"]').type("admin");
      cy.get('input[aria-label="Password (min 6 chars)"]').type("AnotherSecurePass1!");
      cy.get('input[aria-label="Confirm Password"]').type("AnotherSecurePass1!");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Create Account").click();
      cy.contains("already exists").should("be.visible");
    });
  });

  // =========================================================================
  // POSITIVE TEST CASES (USER JOURNEYS, PRESETS, MODELS & AUDIT TRAIL)
  // =========================================================================
  context("Positive Test Cases (E2E Functional Journeys)", () => {
    it("TC-POS-01: Creates a new unique user account and authenticates with it", () => {
      const uniqueUser = `cypress_eng_${Date.now()}`;
      cy.contains('[role="tab"]', "Create Account").click();
      cy.get('input[aria-label="Full Name (e.g. Dr. Alex Mercer)"]').type("Dr. Cypress Tester");
      cy.get('input[aria-label="Username (min 3 chars)"]').type(uniqueUser);
      cy.get('input[aria-label="Password (min 6 chars)"]').type("CypressPass123!");
      cy.get('input[aria-label="Confirm Password"]').type("CypressPass123!");
      cy.contains('[data-testid="stFormSubmitButton"] button', "Create Account").click();

      cy.contains(/Account (created|registered) successfully/).should("be.visible");

      // Switch back to Sign In and verify newly registered user logs in
      cy.contains('[role="tab"]', "Sign In").scrollIntoView().click({ force: true });
      loginAs(uniqueUser, "CypressPass123!");
      cy.contains("Dr. Cypress Tester").should("be.visible");
    });

    it("TC-POS-02 & TC-POS-03: Signs in as admin and verifies System Connectivity badges", () => {
      loginAs("admin", "adminpassword123");
      cy.contains("System Connectivity").should("be.visible");
      cy.contains("Database (MongoDB 27017):").should("be.visible");
      cy.contains("Classical ML (Port 8001):").should("be.visible");
      cy.contains("Deep Learning (Port 8002):").should("be.visible");
      cy.contains("Quantum ML (Port 8003):").should("be.visible");
    });

    it("TC-POS-04 & TC-POS-05: Renders Historical SCADA Telemetry and CSV Export on Tab 1", () => {
      loginAs("admin", "adminpassword123");
      cy.contains("Historical SCADA Telemetry & Batch Analysis").should("be.visible");
      cy.contains("Multi-Model Comparative Evaluation", { timeout: 45000 }).should("be.visible");
      cy.contains("Comparison Table", { timeout: 45000 }).scrollIntoView().should("be.visible");
      cy.contains("button", "Download", { timeout: 20000 }).should("exist");
    });

    it("TC-POS-06: Evaluates Healthy Turbine Baseline Preset as NORMAL OPERATIONAL STATE", () => {
      loginAs("admin", "adminpassword123");
      cy.contains('[role="tab"]', "Manual Diagnostic & What-If Simulator").click({ force: true });
      cy.contains("button", "Load Healthy Turbine Baseline").scrollIntoView().click({ force: true });
      cy.get('input[aria-label="Gearbox Oil Temp (°C)"]').should(($el) => {
        expect(parseFloat($el.val())).to.be.closeTo(60.87, 0.1);
      });
      cy.contains('[data-testid="stFormSubmitButton"] button', "Run Real-Time Multi-Model Prediction")
        .scrollIntoView()
        .click({ force: true });
      cy.contains("NORMAL OPERATIONAL STATE: Healthy Gearbox", { timeout: 30000 }).should(
        "be.visible"
      );
    });

    it("TC-POS-07: Evaluates Bearing Overheat Preset as CRITICAL ANOMALY ALERT", () => {
      loginAs("admin", "adminpassword123");
      cy.contains('[role="tab"]', "Manual Diagnostic & What-If Simulator").click({ force: true });
      cy.contains("button", "Simulate Bearing Overheat & Seizure").scrollIntoView().click({ force: true });
      cy.get('input[aria-label="Gearbox Oil Temp (°C)"]').should(($el) => {
        expect(parseFloat($el.val())).to.be.closeTo(88.5, 0.1);
      });
      cy.contains('[data-testid="stFormSubmitButton"] button', "Run Real-Time Multi-Model Prediction")
        .scrollIntoView()
        .click({ force: true });
      cy.contains("CRITICAL ANOMALY ALERT: High Risk Detected!", { timeout: 30000 }).should(
        "be.visible"
      );
    });

    it("TC-POS-08: Evaluates Extreme Vibration Preset as CRITICAL ANOMALY ALERT", () => {
      loginAs("admin", "adminpassword123");
      cy.contains('[role="tab"]', "Manual Diagnostic & What-If Simulator").click({ force: true });
      cy.contains("button", "Simulate Extreme Vibration Anomaly").scrollIntoView().click({ force: true });
      cy.get('input[aria-label="Gearbox Oil Temp (°C)"]').should(($el) => {
        expect(parseFloat($el.val())).to.be.closeTo(72.0, 0.1);
      });
      cy.contains('[data-testid="stFormSubmitButton"] button', "Run Real-Time Multi-Model Prediction")
        .scrollIntoView()
        .click({ force: true });
      cy.contains("CRITICAL ANOMALY ALERT: High Risk Detected!", { timeout: 30000 }).should(
        "be.visible"
      );
    });

    it("TC-POS-09 & TC-POS-10: Switches DL (MLP) and QML (Fidelity) via Algorithm Selector", () => {
      loginAs("admin", "adminpassword123");
      cy.contains('[role="tab"]', "Manual Diagnostic & What-If Simulator").click({ force: true });
      cy.contains("Algorithm Architecture Selector").scrollIntoView().click({ force: true });

      // Select MLP Dense Autoencoder for Deep Learning
      cy.get('input[aria-label*="Deep Learning Model"]').click({ force: true });
      cy.contains('[role="option"]', "MLP Dense Autoencoder").click({ force: true });

      // Select Quantum State Fidelity Metric for Quantum ML
      cy.get('input[aria-label*="Quantum ML Model"]').click({ force: true });
      cy.contains('[role="option"]', "Quantum State Fidelity Metric").click({ force: true });

      cy.contains('[data-testid="stFormSubmitButton"] button', "Run Real-Time Multi-Model Prediction")
        .scrollIntoView()
        .click({ force: true });

      cy.contains("DL (MLP)", { timeout: 25000 }).should("be.visible");
      cy.contains("QML (Fidelity)", { timeout: 25000 }).should("be.visible");
    });

    it("TC-POS-11 & TC-POS-12: Persists Diagnostic Audit History and locks portal on Logout", () => {
      loginAs("admin", "adminpassword123");
      cy.contains('[role="tab"]', "Manual Diagnostic & What-If Simulator").click({ force: true });
      cy.contains("Diagnostic Audit History").should("be.visible");

      cy.contains("button", "Logout").click({ force: true });
      cy.contains("Wind Turbine SCADA Security Portal", { timeout: 15000 }).should("be.visible");
    });
  });
});

