//
//  SwipeIntroView.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import SwiftUI

struct SwipeIntroView: View {
    @EnvironmentObject var viewModel : MovieViewModel
    @State private var swipeCount = 0
    @State private var goToHome = false
    

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack {
                // TITLE
                Text("Swipe")
                    .font(.title)
                    .foregroundColor(.white)

                // SUBTITLE
                Text("Swipe to build your taste. Swipe left on things you don't like or haven't watched. And right on things you love!")
                    .foregroundColor(.gray)

                Spacer()
                
                
            }

            if viewModel.movies.isEmpty {
                ProgressView()
            } else {
                ZStack {
                    ForEach(Array(viewModel.movies.prefix(5))) { movie in
                        SwipeCard(movie: movie) {
                            remove(movie)
                        }
                    }
                }
            }
        }
            
        
        .onAppear {
            viewModel.fetchMovies()
        }
        .navigationDestination(isPresented: $goToHome){
            HomeView()
        }
    }

    func remove(_ movie: Movie) {
        viewModel.movies.removeAll { $0.id == movie.id }
        
        swipeCount += 1
        
        if swipeCount >= 5 {
            goToHome = true
            
        }
        
    }
}

#Preview {
    SwipeIntroView()
}
