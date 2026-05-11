//
//  ContentView.swift
//  EDEN
//
//  Created by Alana Kumar on 17/4/2026.
//

import SwiftUI

struct ContentView: View {
    @State private var goToWelcome = false

    var body: some View {
        if goToWelcome {
            WelcomeScreen()
        } else {
            ZStack {
                Image("moviebg")
                    .resizable()
                    .scaledToFill()
                    .ignoresSafeArea()

                Color.black
                    .opacity(0.3)
                    .ignoresSafeArea()

                VStack(spacing: 25) {
                    Spacer()

                    Image("edenlogo")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 300)
                        .shadow(color: .purple, radius: 20)

                    Text("Movie Recommendation App")
                        .foregroundColor(.white)
                        .font(.system(size: 22, weight: .bold))

                    ProgressView()
                        .progressViewStyle(
                            CircularProgressViewStyle(tint: .cyan)
                        )
                        .scaleEffect(1.7)

                    Spacer()
                }
                .padding()
            }
            .onAppear {
                // ⏱ wait 1.5 seconds then go to welcome
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    goToWelcome = true
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
