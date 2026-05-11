//
//  HomeView.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import SwiftUI

struct HomeView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 20) {

                Text("E.D.E.N")
                    .font(.largeTitle)
                    .foregroundColor(.purple)

                Text("Your movie space 🎬")
                    .foregroundColor(.white)

                Spacer()

                // Example movie cards
                ForEach(0..<5) { _ in
                    RoundedRectangle(cornerRadius: 15)
                        .fill(Color.gray.opacity(0.3))
                        .frame(height: 150)
                        .padding(.horizontal)
                }

                Spacer()
            }
        }
    }
}
