//
//  WelcomeScreen.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import SwiftUI

struct WelcomeScreen: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 25) {

                Image("edenlogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 420, height: 300)

                Image("welcomeImage")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 420)

                Text("Welcome to E.D.E.N")
                    .font(.title)
                    .foregroundColor(.white)

                Text("Rate and review movies then discover movies you'll love")
                    .foregroundColor(.gray)

                Spacer()

                NavigationLink(destination: LoginView()) {
                    Text("Get Started")
                        .fontWeight(.semibold)
                        .foregroundColor(.black)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(
                            LinearGradient(
                                colors: [.purple, .blue],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .cornerRadius(12)
                }
                .padding(.horizontal)

                Spacer()
            }
        }
    }
}
#Preview {
    WelcomeScreen()
}
