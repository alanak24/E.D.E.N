//
//  LoginView.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import SwiftUI

struct LoginView: View {
    @State private var username = ""
    @State private var email = ""
    @State private var password = ""
    var body: some View {
        ZStack {
            // Background\
            Color.black.ignoresSafeArea()
            VStack(spacing: 20) {

                Spacer()

                // Cute icon (your popcorn)
                Image("tvcharacters")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 220)

                Text("Create your account")
                    .font(.title)
                    .foregroundColor(.white)

                // Buttons
                VStack(spacing: 15) {

                    Button("Continue with Facebook") {
                        // later
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        LinearGradient(
                            colors: [.purple, .blue],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .foregroundColor(.white)
                    .cornerRadius(12)

                    Button("Continue with Google") {
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        LinearGradient(
                            colors: [.purple, .blue],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .foregroundColor(.white)
                    .cornerRadius(12)
                    
                }
                VStack(spacing: 12) {

                    // Username
                    ZStack(alignment: .leading) {
                        if username.isEmpty {
                            Text("Username")
                                .foregroundColor(.white.opacity(0.6))
                                .padding(.horizontal)
                        }

                        TextField("", text: $username)
                            .foregroundColor(.white)
                            .padding()
                    }
                    .background(Color.white.opacity(0.1))
                    .cornerRadius(10)

                    // Email
                    ZStack(alignment: .leading) {
                        if email.isEmpty {
                            Text("Email")
                                .foregroundColor(.white.opacity(0.6))
                                .padding(.horizontal)
                        }

                        TextField("", text: $email)
                            .foregroundColor(.white)
                            .padding()
                    }
                    .background(Color.white.opacity(0.1))
                    .cornerRadius(10)

                    // Password
                    ZStack(alignment: .leading) {
                        if password.isEmpty {
                            Text("Password")
                                .foregroundColor(.white.opacity(0.6))
                                .padding(.horizontal)
                        }

                        SecureField("", text: $password)
                            .foregroundColor(.white)
                            .padding()
                    }
                    .background(Color.white.opacity(0.1))
                    .cornerRadius(10)
                }

                // MARK: - Get Started Button

                NavigationLink(destination: SwipeIntroView()) {
                    Text("Get Swiping")
                        .foregroundColor(.black)
                        .fontWeight(.semibold)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.white)
                        .cornerRadius(12)
                }
                

            }
            .padding()
        }
    }
}

#Preview {
    LoginView()
}
