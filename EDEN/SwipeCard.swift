//
//  SwipeCard.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import SwiftUI

struct SwipeCard: View {
    @EnvironmentObject var movieVM: MovieViewModel
    @State private var showLike = false
    @State private var showDislike = false
    let movie: Movie
    var onRemove: () -> Void
    
    
    @State private var offset = CGSize.zero
    
    var body: some View {
        ZStack {
            
            // MAIN CARD
            VStack {
                AsyncImage(
                    url: URL(string: "https://image.tmdb.org/t/p/w500\(movie.poster_path)")
                ) { image in
                    image.resizable().scaledToFit()
                } placeholder: {
                    ProgressView()
                }
                .frame(height: 400)
                .cornerRadius(20)
                
                Text(movie.title)
                    .foregroundColor(.white)
                    .font(.headline)
            }
            .background(Color.black.opacity(0.8))
            .cornerRadius(20)
            
            //  LIKE
            if showLike {
                Text("❤️")
                    .font(.system(size: 80))
                    .opacity(0.8)
                    .transition(.scale)
            }
            
            // DISLIKE
            if showDislike {
                Text("❌")
                    .font(.system(size: 80))
                    .opacity(0.8)
                    .transition(.scale)
            }
        }
        .offset(x: offset.width)
        .rotationEffect(.degrees(Double(offset.width) / 20))
        .gesture(
            DragGesture()
                .onChanged { value in
                    offset = value.translation
                }
                .onEnded { _ in
                    if offset.width > 100 {
                        showLike = true
                        
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            onRemove()
                            showLike = false
                        }
                        
                    } else if offset.width < -100 {
                        showDislike = true
                        
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                            onRemove()
                            showDislike = false
                        }
                    }
                    
                    offset = .zero
                }
        )
        .animation(.easeInOut, value: showLike)
        .animation(.easeInOut, value: showDislike)
        .animation(.spring(), value: offset)
    }
}
